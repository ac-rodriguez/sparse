import tensorflow as tf
import sys
from utils import inv_preprocess, decode_labels_reg
from colorize import colorize, inv_preprocess_tf

def bn_layer(X, activation_fn, is_training=True):
    return tf.contrib.layers.batch_norm(
        X,
        activation_fn=activation_fn,
        is_training=is_training,
        updates_collections=None,
        scale=True,
        scope=None)


log10 = lambda x: tf.log(x) / tf.log(10.0)


def s2n(a, b):
    sn = tf.reduce_mean(tf.squared_difference(a, b))
    sn = 10 * log10(255.0 / sn)

    return sn


def snr_metric(a, b):
    sd, sd_op = tf.metrics.mean_squared_error(a, b)

    s2n = 10 * log10(255.0 / sd)

    return s2n, sd_op


def resid_block(X, filters=[64, 128], only_resid=False):
    Xr = tf.layers.conv2d(X, filters=filters[0], kernel_size=3, activation=tf.nn.relu, padding='same')
    Xr = bn_layer(Xr, tf.nn.relu)
    Xr = tf.layers.conv2d(Xr, filters=filters[1], kernel_size=1, activation=tf.nn.relu, padding='same')
    Xr = bn_layer(Xr, tf.nn.relu)
    if only_resid:
        return Xr
    else:
        return X + Xr


def resid_block1(X, filters=[64, 128], only_resid=False, scale=0.1):
    Xr = tf.layers.conv2d(X, filters=filters[0], kernel_size=3, activation=tf.nn.relu, padding='same')
    # Xr = bn_layer(Xr, tf.nn.relu)
    Xr = tf.layers.conv2d(Xr, filters=filters[1], kernel_size=1, activation=tf.nn.relu, padding='same')

    Xr = Xr * scale

    if only_resid:
        return Xr
    else:
        return X + Xr


def deeplab(input, n_channels):
    with tf.variable_scope('resnet_blocks'):
        x = resid_block(input, filters=[128, 128], only_resid=True)
        for i in range(6):
            x = resid_block(x)

    hr_hat = tf.layers.conv2d(x, filters=n_channels, kernel_size=3, activation=None,
                              padding='same')
    return hr_hat


def deep_sentinel2(input, n_channels, is_resid=True):
    feature_size = 128
    with tf.variable_scope('resnet_blocks'):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x = tf.layers.conv2d(input, feature_size, kernel_size=3, activation=tf.nn.relu, padding='same')
        for i in range(6):
            # features_nn = resid_block(features_nn)
            x = resid_block1(x, filters=[feature_size, feature_size])
            # features_nn = resid_block(features_nn)

    hr_hat = tf.layers.conv2d(x, filters=n_channels, kernel_size=3, activation=None,
                              padding='same')
    if is_resid:
        return hr_hat + input[..., 0:3]
    else:
        return hr_hat


def model_fn(features, labels, mode, params={}):
    args = params['args']
    feat_l, feat_h = features['feat_l'], features['feat_h']
    feat_h_down = tf.image.resize_bilinear(feat_h, size=[args.patch_size, args.patch_size], name='HR_downsampled')
    lab_down = tf.image.resize_bilinear(labels, size=[args.patch_size, args.patch_size], name='Label_downsampled')
    # features = features[...,0:args.n_channels]
    if args.model == '1':  # Baseline
        feat = tf.concat([feat_h_down, feat_l], axis=3)
        y_hat = deep_sentinel2(feat, n_channels=1, is_resid=False)

    elif args.model == '2':
        x_h = tf.layers.conv2d(feat_h_down, 128, 3, activation=tf.nn.relu, padding='same')
        x_l = tf.layers.conv2d(feat_l, 128, 3, activation=tf.nn.relu, padding='same')
        feat = tf.concat([x_h, x_l], axis=3)
        y_hat = deep_sentinel2(feat, n_channels=1, is_resid=False)

    else:
        print('Model {} not defined'.format(args.model))
        sys.exit(1)

    graph = tf.get_default_graph()
    mean_train = graph.get_tensor_by_name("mean_train:0")
    scale = graph.get_tensor_by_name("scale_preprocessing:0")


    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'y_hat': y_hat}

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # loss = tf.cond(flat_(tf.greater_equal(lab_down,0)),lambda: flat_(tf.reduce_sum(tf.abs(y_hat - lab_down))), lambda: tf.zeros_like(flat_(y_hat)))
    cond_dif = tf.where(tf.greater_equal(lab_down, 0),
                        tf.abs(y_hat - lab_down),  ## for wherever i have labels
                        tf.zeros_like(y_hat))  ## semi-supervised loss TODO
    loss = tf.reduce_sum(cond_dif)

    is_summaries = True
    if is_summaries and not args.is_multi_gpu:

        mean_rgb = mean_train  # [..., 0:3]

        # Ploting only rgb and transforming from bgr to rgb.
        inv_ = lambda x: tf.py_func(inv_preprocess, [x, args.batch_size, mean_rgb, scale],
                                    tf.uint8,
                                    name='inv_preprocess_image_rgb')

        inv_reg_ = lambda x: tf.py_func(decode_labels_reg, [x, args.batch_size], tf.uint8,
                                        name='decode_labels_r')
        uint8_ = lambda x: tf.cast(x * 255.0, dtype=tf.uint8)

        image_array_top = tf.concat(axis=2, values=[inv_reg_(lab_down), inv_reg_(y_hat)])
        image_array_mid = tf.concat(axis=2,
                                    values=[inv_(feat_l), uint8_(feat_h_down)])

        image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid])

        tf.summary.image('all',
                         image_array,
                         max_outputs=2)
    else:
        mean_rgb = mean_train  # [..., 0:3]
        uint8_ = lambda x: tf.cast(x * 255.0, dtype=tf.uint8)
        inv_ = lambda x: inv_preprocess_tf(x, mean_rgb, scale_luminosity=scale)

        inv_reg_ = lambda x: uint8_(colorize(x,vmin=0,vmax=4,cmap='hot'))


        image_array_top = tf.concat(axis=2, values=[tf.map_fn(inv_reg_, lab_down,dtype=tf.uint8),
                                                    tf.map_fn(inv_reg_,y_hat,dtype=tf.uint8)])
        a = tf.map_fn(inv_,feat_l, dtype=tf.uint8)
        b = uint8_(feat_h_down)

        image_array_mid = tf.concat(axis=2, values=[a, b])

        image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid])

        tf.summary.image('all',
                         image_array,
                         max_outputs=2)



    if mode == tf.estimator.ModeKeys.TRAIN:
        tf.summary.scalar('metrics/mse', tf.reduce_mean(tf.squared_difference(lab_down, y_hat)))
        tf.summary.scalar('metrics/mae', tf.reduce_mean(tf.abs(lab_down - y_hat)))

        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Compute evaluation metrics.
    eval_metric_ops = {
        'metrics/mae': tf.metrics.mean_absolute_error(
            labels=lab_down, predictions=y_hat),
        'metrics/mse': tf.metrics.mean_squared_error(
            labels=lab_down, predictions=y_hat)}

    # Add summary hook for image summary
    eval_summary_hook = tf.train.SummarySaverHook(
        save_steps=200,
        output_dir=params['model_dir'] + '/eval',
        summary_op=tf.summary.merge_all())  # tf.get_collection('Images')))

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_summary_hook])
