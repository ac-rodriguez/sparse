import tensorflow as tf
import sys
from utils import inv_preprocess, decode_labels_reg
from colorize import colorize, inv_preprocess_tf, slice_last_dim
import numpy as np


def bn_layer(X, activation_fn=None, is_training=True):
    if activation_fn is None:
        activation_fn = lambda x: x
    return activation_fn(tf.layers.batch_normalization(X, training=is_training))



log10 = lambda x: tf.log(x) / tf.log(10.0)


def s2n(a, b):
    sn = tf.reduce_mean(tf.squared_difference(a, b))
    sn = 10 * log10(255.0 / sn)

    return sn


def snr_metric(a, b):
    sd, sd_op = tf.metrics.mean_squared_error(a, b)

    s2n = 10 * log10(255.0 / sd)

    return s2n, sd_op


def resid_block(X, filters=[64, 128], is_residual=True, is_training=True, is_batch_norm=True):
    Xr = tf.layers.conv2d(X, filters=filters[0], kernel_size=3, activation=tf.nn.relu, padding='same')
    if is_batch_norm:
        Xr = bn_layer(Xr, tf.nn.relu, is_training=is_training)
    Xr = tf.layers.conv2d(Xr, filters=filters[1], kernel_size=1, activation=tf.nn.relu, padding='same')
    if is_batch_norm:
        Xr = bn_layer(Xr, tf.nn.relu, is_training=is_training)
    if is_residual:
        Xr = X + Xr
    return X + Xr


def sum_pool(X, scale, name):
    return tf.multiply(float(scale),
                       tf.nn.avg_pool(X, ksize=(1, scale, scale, 1),
                                      strides=(1, scale, scale, 1), padding='VALID'),
                       name=name)
def bilinear(X,size, name=None):
    return tf.image.resize_bilinear(X, size=[int(size), int(size)], name=name)



def resid_block1(X, filters=[64, 128], is_residual=False, scale=0.1):
    Xr = tf.layers.conv2d(X, filters=filters[0], kernel_size=3, activation=tf.nn.relu, padding='same')
    # Xr = bn_layer(Xr, tf.nn.relu)
    Xr = tf.layers.conv2d(Xr, filters=filters[1], kernel_size=1, activation=tf.nn.relu, padding='same')

    Xr = Xr * scale

    if is_residual:
        return X + Xr
    else:
        return Xr


def deeplab(input, n_channels, is_training=True):
    with tf.variable_scope('resnet_blocks'):
        x = resid_block(input, filters=[128, 128], is_residual=False, is_training=is_training)
        for i in range(6):
            x = resid_block(x, is_training=is_training)

    hr_hat = tf.layers.conv2d(x, filters=n_channels, kernel_size=3, activation=None,
                              padding='same')
    return hr_hat


def deep_sentinel2(input, n_channels, is_residual=True, scope_name='resnet_blocks', is_training=True,
                   is_batch_norm=True):
    feature_size = 128
    with tf.variable_scope(scope_name):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x = tf.layers.conv2d(input, feature_size, kernel_size=3, activation=tf.nn.relu, padding='same')
        for i in range(6):
            # features_nn = resid_block(features_nn)
            x = resid_block(x, filters=[feature_size, feature_size], is_residual=True, is_training=is_training,
                            is_batch_norm=is_batch_norm)
            # features_nn = resid_block(features_nn)

    hr_hat = tf.layers.conv2d(x, filters=n_channels, kernel_size=3, activation=None,
                              padding='same')
    if is_residual:
        return hr_hat + input
    else:
        return hr_hat


def SR_task(feat_l, args,is_batch_norm=True, is_training=True):
    with tf.variable_scope('SR_task'):
        feat_l_up = bilinear(feat_l,size=args.patch_size*args.scale,name='LR_up')

        HR_hat1 = deep_sentinel2(feat_l_up, n_channels=3, is_residual=False, is_training=is_training,
                                 is_batch_norm=is_batch_norm)

        feat_l_rgb = slice_last_dim(feat_l_up, dims=(2, 1, 0))

        HR_hat1 = tf.nn.sigmoid(HR_hat1) + bn_layer(feat_l_rgb, is_training=is_training)

        HR_hat = tf.layers.conv2d(HR_hat1, 3, 3, activation=tf.nn.sigmoid, padding='same')

        return HR_hat


def model_fn(features, labels, mode, params={}):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    graph = tf.get_default_graph()
    try:
        mean_train = graph.get_tensor_by_name("mean_train:0")
        scale = graph.get_tensor_by_name("scale_preprocessing:0")

    except KeyError:
        mean_train = tf.Variable(np.zeros(11), name='mean_train', trainable= False, dtype=tf.float32)
        scale = tf.Variable(10.0, name='scale_preprocessing', trainable = False, dtype=tf.float32)

    args = params['args']
    if isinstance(features, dict):
        feat_l, feat_h = features['feat_l'], features['feat_h']
    else:
        feat_l = features
        feat_h = None


    if args.model == '1':  # Baseline  No High Res for training
        y_hat = deep_sentinel2(feat_l, n_channels=1, is_residual=False, is_training=is_training, is_batch_norm=True)

    elif args.model == '1a':  # Baseline  No High Res for training without BN
        y_hat = deep_sentinel2(feat_l, n_channels=1, is_residual=False, is_training=is_training, is_batch_norm=False)

    elif args.model == '2':  # SR as a side task
        HR_hat = SR_task(feat_l=feat_l + mean_train, args=args,is_batch_norm=True, is_training=is_training)

        HR_hat_down = bilinear(HR_hat, args.patch_size, name='HR_hat_down')

        # Estimated sup-pixel features from LR
        x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

        x_l = tf.layers.conv2d(feat_l, 128, 3, activation=tf.nn.relu, padding='same')

        feat = tf.concat([x_h, x_l], axis=3)

        y_hat = deep_sentinel2(feat, n_channels=1, is_residual=False, is_training=is_training, is_batch_norm=True)
    elif args.model == '2a':  # SR as a side task without BN
        HR_hat = SR_task(feat_l=feat_l + mean_train, args=args,is_batch_norm=False, is_training=is_training)

        HR_hat_down = bilinear(HR_hat, args.patch_size, name='HR_hat_down')

        # Estimated sup-pixel features from LR
        x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

        x_l = tf.layers.conv2d(feat_l, 128, 3, activation=tf.nn.relu, padding='same')

        feat = tf.concat([x_h, x_l], axis=3)

        y_hat = deep_sentinel2(feat, n_channels=1, is_residual=False, is_training=is_training, is_batch_norm=False)

    elif args.model == '3':  # SR as a side task - leaner version
        HR_hat = SR_task(feat_l=feat_l + mean_train, args=args, is_training=is_training, is_batch_norm=True)

        HR_hat_down = bilinear(HR_hat, args.patch_size, name='HR_hat_down')

        # Estimated sup-pixel features from LR
        y_hat = deep_sentinel2(HR_hat_down, n_channels=1, is_residual=False, is_training=is_training)

    else:
        print('Model {} not defined'.format(args.model))
        sys.exit(1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y_hat': y_hat}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    lab_down = sum_pool(labels, args.scale, name='Label_down')
    feat_h_down = bilinear(feat_h, args.patch_size, name='HR_down')

    uint8_ = lambda x: tf.cast(x * 255.0, dtype=tf.uint8)
    inv_ = lambda x: inv_preprocess_tf(x, mean_train, scale_luminosity=scale)
    inv_reg_ = lambda x: uint8_(colorize(x, vmin=0, vmax=4, cmap='hot'))

    image_array_top = tf.concat(axis=2, values=[tf.map_fn(inv_reg_, lab_down, dtype=tf.uint8),
                                                tf.map_fn(inv_reg_, y_hat, dtype=tf.uint8)])
    a = tf.map_fn(inv_, feat_l, dtype=tf.uint8)
    b = uint8_(feat_h_down)

    image_array_mid = tf.concat(axis=2, values=[a, b])

    image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid])

    tf.summary.image('all',
                     image_array,
                     max_outputs=2)

    if not '1' in args.model:
        feat_l_up = tf.map_fn(inv_,
                              bilinear(feat_l, size=args.patch_size * args.scale), dtype=tf.uint8)
        image_array = tf.concat(axis=2,
                                values=[feat_l_up, uint8_(HR_hat), uint8_(feat_h)])

        tf.summary.image('HR_hat-HR',
                         image_array,
                         max_outputs=2)
        semi_loss = tf.nn.l2_loss(HR_hat - feat_h)
    else:
        semi_loss = 0


    superv_loss = tf.reduce_sum(tf.where(tf.greater_equal(lab_down, 0),
                                         tf.abs(y_hat - lab_down),  ## for wherever i have labels
                                         tf.zeros_like(y_hat)))  ## ignoring whenever I don't have labels

    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'kernel' in v.name]

    loss = superv_loss + args.lambda_loss * semi_loss + tf.reduce_sum(l2_losses)






    # train_hook = tf.train.SessionRunHook().after_run()
    if mode == tf.estimator.ModeKeys.TRAIN:
        # tf.summary.scalar('metrics/iou', iou)
        # tf.summary.scalar('metrics/mse', tf.reduce_mean(tf.squared_difference(lab_down, y_hat)))
        # tf.summary.scalar('metrics/mae', tf.reduce_mean(tf.abs(lab_down - y_hat)))
        # tf.summary.scalar('metrics/mae', tf.reduce_mean(tf.abs(lab_down - y_hat)))

        tf.summary.scalar('metrics/superv_loss', superv_loss)
        tf.summary.scalar('metrics/semi_loss', semi_loss)

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Semantic Label
    label_sem = tf.greater_equal(lab_down, 1)
    y_hat_sem = tf.greater_equal(y_hat, 1)


    # Compute evaluation metrics.
    eval_metric_ops = {
        'metrics/mae': tf.metrics.mean_absolute_error(
            labels=lab_down, predictions=y_hat),
        'metrics/mse': tf.metrics.mean_squared_error(
            labels=lab_down, predictions=y_hat),
        'metrics/iou': tf.metrics.mean_iou(label_sem,y_hat_sem, num_classes=2),
        'metrics/prec': tf.metrics.precision(label_sem, y_hat_sem, num_classes=2),
        'metrics/recall': tf.metrics.recall(label_sem, y_hat_sem, num_classes=2)}

    if not '1' in args.model:
        eval_metric_ops['metrics/semi_loss'] = tf.metrics.mean_squared_error(HR_hat, feat_h)

    # Add summary hook for image summary
    eval_summary_hook = tf.train.SummarySaverHook(
        save_steps=200,
        output_dir=params['model_dir'] + '/eval',
        summary_op=tf.summary.merge_all())  # tf.get_collection('Images')))

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_summary_hook]) #, logging_hook])
