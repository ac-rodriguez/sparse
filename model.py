import tensorflow as tf
import sys
import numpy as np

from colorize import colorize, inv_preprocess_tf
from models_reg import simple
from models_sr import SR_task, dbpn_SR
from tools_tf import bilinear, snr_metric


def summaries(feat_h, feat_l, labels, label_sem, w, y_hat, HR_hat, is_SR, args, is_training):
    graph = tf.get_default_graph()
    try:
        mean_train = graph.get_tensor_by_name("mean_train_k:0")
        scale = graph.get_tensor_by_name("std_train_k:0")
    except KeyError:
        # if constants are not defined in the graph yet,
        # after loading pre-trained networks tf.Variables are used with the correct values
        mean_train = tf.Variable(np.zeros(11), name='mean_train', trainable=False, dtype=tf.float32)
        scale = tf.Variable(np.ones(11), name='std_train', trainable=False, dtype=tf.float32)


    uint8_ = lambda x: tf.cast(x * 255.0, dtype=tf.uint8)
    inv_ = lambda x: inv_preprocess_tf(x, mean_train, scale_luminosity=scale)
    inv_reg_ = lambda x: uint8_(colorize(x, vmin=0, vmax=4, cmap='hot'))
    inv_sem_ = lambda x: uint8_(colorize(x, vmin=0, vmax=1, cmap='hot'))
    int_ = lambda x: tf.cast(x, dtype=tf.int64)

    pred_class = int_(tf.argmax(y_hat['sem'], axis=3))

    image_array_top = tf.concat(axis=2, values=[tf.map_fn(inv_reg_, labels, dtype=tf.uint8),
                                                tf.map_fn(inv_reg_, y_hat['reg'], dtype=tf.uint8)])

    image_array_mid = tf.concat(axis=2, values=[tf.map_fn(inv_sem_, label_sem, dtype=tf.uint8),
                                                tf.map_fn(inv_sem_, pred_class, dtype=tf.uint8)])

    a_ = tf.map_fn(inv_, feat_l, dtype=tf.uint8)
    feat_h_down = uint8_(bilinear(feat_h, args.patch_size, name='HR_down')) if feat_h is not None else a_

    image_array_bottom = tf.concat(axis=2, values=[a_, feat_h_down])  # , uint8_(feat_h_down)])

    image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

    tf.summary.image('all',
                     image_array,
                     max_outputs=2)

    if is_SR:
        feat_l_up = tf.map_fn(inv_,
                              bilinear(feat_l, size=args.patch_size * args.scale), dtype=tf.uint8)
        image_array = tf.concat(axis=2,
                                values=[feat_l_up, uint8_(HR_hat), uint8_(feat_h)])

        tf.summary.image('HR_hat-HR',
                         image_array,
                         max_outputs=2)


    if not is_training:

        # Compute evaluation metrics.
        eval_metric_ops = {
            'metrics/mae': tf.metrics.mean_absolute_error(
                labels=labels, predictions=y_hat['reg'], weights=w),
            'metrics/mse': tf.metrics.mean_squared_error(
                labels=labels, predictions=y_hat['reg'], weights=w),
            'metrics/iou': tf.metrics.mean_iou(labels=label_sem, predictions=pred_class, num_classes=2, weights=w),
            'metrics/prec': tf.metrics.precision(labels=label_sem, predictions=pred_class, weights=w),
            'metrics/acc': tf.metrics.accuracy(labels=label_sem, predictions=pred_class, weights=w),
            'metrics/recall': tf.metrics.recall(labels=label_sem, predictions=pred_class, weights=w)}

        if is_SR:
            eval_metric_ops['metrics/semi_loss'] = tf.metrics.mean_squared_error(HR_hat, feat_h)
            eval_metric_ops['metrics/s2nr'] = snr_metric(HR_hat, feat_h)
    else:
        eval_metric_ops = None

    return eval_metric_ops


def model_fn(features, labels, mode, params={}):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    args = params['args']
    if isinstance(features, dict):
        feat_l, feat_h = features['feat_l'], features['feat_h']
    else:
        feat_l = features
        feat_h = None

    if args.is_bilinear:
        down_ = lambda x: bilinear(x,args.patch_size,name='HR_hat_down')
    else:
        down_ = lambda x: tf.layers.conv2d(x,3,3,strides=args.scale,padding='same')
    args.patch_size = feat_l.shape[1]
    is_SR = True
    if args.model == 'simple':  # Baseline  No High Res for training
        y_hat = simple(feat_l, n_channels=1, is_training=is_training)
        is_SR = False
        HR_hat = None
    elif args.model == 'simple2':  # SR as a side task
        HR_hat = SR_task(feat_l=feat_l, args=args, is_batch_norm=True, is_training=is_training)

        HR_hat_down = down_(HR_hat)
        # HR_hat_down = tf.layers.average_pooling2d(HR_hat,args.scale,args.scale)

        # Estimated sub-pixel features from LR
        x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

        x_l = tf.layers.conv2d(feat_l, 128, 3, activation=tf.nn.relu, padding='same')

        feat = tf.concat([x_h, x_l], axis=3)

        y_hat = simple(feat, n_channels=1, is_training=is_training)
    elif args.model == 'simple2a':  # SR as a side task
        HR_hat = SR_task(feat_l=feat_l, args=args, is_batch_norm=False, is_training=is_training)

        HR_hat_down = down_(HR_hat)

        # Estimated sup-pixel features from LR
        x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

        x_l = tf.layers.conv2d(feat_l, 128, 3, activation=tf.nn.relu, padding='same')

        feat = tf.concat([x_h, x_l], axis=3)

        y_hat = simple(feat, n_channels=1, is_training=is_training)
    elif args.model == 'simple3':  # SR as a side task - leaner version
        HR_hat = SR_task(feat_l=feat_l, args=args, is_training=is_training, is_batch_norm=True)

        HR_hat_down = down_(HR_hat)

        # Estimated sup-pixel features from LR
        y_hat = simple(HR_hat_down, n_channels=1, is_training=is_training)

    elif args.model == 'simple3a':  # SR as a side task - leaner version
        HR_hat = SR_task(feat_l=feat_l, args=args, is_training=is_training, is_batch_norm=False)

        HR_hat_down = down_(HR_hat)

        # Estimated sup-pixel features from LR
        y_hat = simple(HR_hat_down, n_channels=1, is_training=is_training)
    elif args.model == 'simple4':  # SR as a side task - leaner version
        HR_hat = dbpn_SR(feat_l=feat_l, is_training=is_training)

        HR_hat_down = down_(HR_hat)

        # Estimated sup-pixel features from LR
        y_hat = simple(HR_hat_down, n_channels=1, is_training=is_training)
    else:
        print('Model {} not defined'.format(args.model))
        sys.exit(1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = y_hat

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    int_ = lambda x: tf.cast(x, dtype=tf.int64)

    # labels = sum_pool(labels, args.scale, name='Label_down')
    # labels = labels
    label_sem = tf.squeeze(int_(tf.greater(labels, 0.5)), axis=3)
    float_ = lambda x: tf.cast(x, dtype=tf.float32)

    w = float_(tf.where(tf.greater_equal(labels, 0.0),
                        tf.ones_like(labels),  ## for wherever i have labels
                        tf.zeros_like(labels)))

    loss_reg = tf.losses.mean_squared_error(labels=labels, predictions=y_hat['reg'], weights=w)

    loss_sem = tf.losses.sparse_softmax_cross_entropy(labels=label_sem, logits=y_hat['sem'], weights=w)

    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if
                 ('weights' in v.name or 'kernel' in v.name)]

    if is_SR:
        loss_sr = tf.nn.l2_loss(HR_hat - feat_h)
    else:
        loss_sr = 0

    loss = args.lambda_reg * loss_reg + (1.0 - args.lambda_reg) * loss_sem + args.lambda_sr * loss_sr + tf.add_n(
        l2_losses)

    tf.summary.scalar('loss/reg', loss_reg)
    tf.summary.scalar('loss/sem', loss_sem)
    tf.summary.scalar('loss/SR', loss_sr)

    eval_metric_ops = summaries(feat_h, feat_l, labels, label_sem, w, y_hat, HR_hat, is_SR, args, is_training)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Add summary hook for image summary
    eval_summary_hook = tf.train.SummarySaverHook(
        save_steps=200,
        output_dir=params['model_dir'] + '/eval',
        summary_op=tf.summary.merge_all())

    return tf.estimator.EstimatorSpec(
        mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_summary_hook])
