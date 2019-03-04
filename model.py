import tensorflow as tf
import sys
import numpy as np

from colorize import colorize, inv_preprocess_tf
from models_reg import simple
from models_sr import SR_task, dbpn_SR, slice_last_dim
from tools_tf import bilinear, snr_metric, sum_pool,avg_pool, get_lr_ADAM

colormax = {2: 0.93, 4: 0.155, 8: 0.04}

class Model:
    def __init__(self, params):
        self.args = params['args']
        self.model_dir = params['model_dir']

        self.scale = self.args.scale
        self.is_hr_label = self.args.is_hr_label
        if self.args.is_bilinear:
            self.down_ = lambda x, _: bilinear(x,self.patch_size, name='HR_hat_down')
            self.up_ = lambda x, _: bilinear(x, self.patch_size * self.scale, name='HR_hat_down')
        else:
            self.down_ = lambda x, ch: tf.layers.conv2d(x, ch, 3, strides=self.scale, padding='same')
            self.up_ = lambda x, ch: tf.layers.conv2d_transpose(x, ch, 3, strides=self.scale, padding='same')

        self.float_ = lambda x: tf.cast(x, dtype=tf.float32)
        self.loss_in_HR = False
        self.sem_threshold = 0.5
        self.model = self.args.model

    def model_fn(self,features, labels, mode):
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # self.features = features
        self.labels = labels

        if isinstance(features, dict):
            self.feat_l, self.feat_h = features['feat_l'], features['feat_h']
        else:
            self.feat_l = features
            self.feat_h = None

        self.patch_size = self.feat_l.shape[1]

        y_hat, HR_hat = self.get_predicitons()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=y_hat)
        if self.loss_in_HR:
            self.sem_threshold = 1e-5
        self.compute_loss(y_hat,HR_hat)
        self.compute_summaries(y_hat,HR_hat)

        if mode == tf.estimator.ModeKeys.TRAIN:
            self.compute_train_op()
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op)

        # Add summary hook for image summary
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=200,
            output_dir=self.model_dir + '/eval',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode, loss=self.loss, eval_metric_ops=self.eval_metric_ops, evaluation_hooks=[eval_summary_hook])

    def get_predicitons(self):
        HR_hat=None
        size=self.patch_size*self.args.scale
        if self.model == 'simple':  # Baseline  No High Res for training
            y_hat = simple(self.feat_l, n_channels=1, is_training=self.is_training)

        elif self.model == 'simplebn':  # Baseline  No High Res for training
            y_hat = simple(self.feat_l, n_channels=1, is_training=self.is_training, is_bn=False)
        elif self.model == 'simple2':  # SR as a side task
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)

            HR_hat_down = self.down_(HR_hat, 3)
            # HR_hat_down = tf.layers.average_pooling2d(HR_hat,args.scale,args.scale)

            # Estimated sub-pixel features from LR
            x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

            x_l = tf.layers.conv2d(self.feat_l, 128, 3, activation=tf.nn.relu, padding='same')

            feat = tf.concat([x_h, x_l], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
        elif self.model == 'simple2a':  # SR as a side task
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_batch_norm=False, is_training=self.is_training)

            HR_hat_down = self.down_(HR_hat, 3)

            # Estimated sup-pixel features from LR
            x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

            x_l = tf.layers.conv2d(self.feat_l, 128, 3, activation=tf.nn.relu, padding='same')

            feat = tf.concat([x_h, x_l], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
        elif self.model == 'simple2b':  # SR as a side task
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)

            HR_hat_down = self.down_(HR_hat, 3)

            feat = tf.concat([HR_hat_down, self.feat_l[..., 3:]], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
        elif self.model == 'simple2c':  # SR as a side task
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)

            feat_l_up = self.up_(self.feat_l[..., 3:], 8)

            # Estimated sub-pixel features from LR
            feat = tf.concat([HR_hat, feat_l_up], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
            for key, val in y_hat.iteritems():
                y_hat[key] = sum_pool(val, self.scale)

        elif self.model == 'simple2d':  # SR as a side task
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)

            feat_l_up = self.up_(self.feat_l, 8)

            # Estimated sub-pixel features from LR
            feat = tf.concat([HR_hat, feat_l_up], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
            if not self.is_hr_label:
                for key, val in y_hat.iteritems():
                    y_hat[key] = bilinear(val, self.patch_size)
            else:
                self.loss_in_HR = True

        elif self.model == 'simple3':  # SR as a side task - leaner version
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_training=self.is_training, is_batch_norm=True)

            HR_hat_down = self.down_(HR_hat, 3)

            # Estimated sup-pixel features from LR
            y_hat = simple(HR_hat_down, n_channels=1, is_training=self.is_training)

        elif self.model == 'simple3a':  # SR as a side task - leaner version
            HR_hat = SR_task(feat_l=self.feat_l, size=size, is_training=self.is_training, is_batch_norm=False)

            HR_hat_down = self.down_(HR_hat, 3)

            # Estimated sup-pixel features from LR
            y_hat = simple(HR_hat_down, n_channels=1, is_training=self.is_training)
        elif self.model == 'simple4':  # SR as a side task - leaner version
            HR_hat = dbpn_SR(feat_l=self.feat_l, is_training=self.is_training, scale=self.args.scale)

            HR_hat_down = self.down_(HR_hat, 3)

            # Estimated sup-pixel features from LR
            y_hat = simple(HR_hat_down, n_channels=1, is_training=self.is_training)
        elif self.model == 'simple4a':  # SR as a side task
            HR_hat = dbpn_SR(feat_l=self.feat_l, is_training=self.is_training, scale=self.args.scale, deep=0)

            HR_hat_down = self.down_(HR_hat, 3)

            # Estimated sub-pixel features from LR
            x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

            x_l = tf.layers.conv2d(self.feat_l, 128, 3, activation=tf.nn.relu, padding='same')

            feat = tf.concat([x_h, x_l], axis=3)
            y_hat = simple(feat, n_channels=1, is_training=self.is_training)

        elif self.model == 'simple4b':  # SR as a side task
            HR_hat = dbpn_SR(feat_l=self.feat_l, is_training=self.is_training, scale=self.args.scale, deep=0)

            HR_hat_down = self.down_(HR_hat, 3)

            # Estimated sub-pixel features from LR
            x_h = tf.layers.conv2d(HR_hat_down, 128, 3, activation=tf.nn.relu, padding='same')

            x_l = tf.layers.conv2d(self.feat_l, 128, 3, activation=tf.nn.relu, padding='same')

            feat = tf.concat([x_h, x_l], axis=3)
            y_hat = simple(feat, n_channels=1, is_training=self.is_training)

        elif self.model == 'simpleHR':
            feat_l_up = self.up_(self.feat_l, 8)

            # Estimated sub-pixel features from LR
            feat = tf.concat([self.feat_h, feat_l_up[..., 3:]], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
            if not self.is_hr_label:
                for key, val in y_hat.iteritems():
                    y_hat[key] = sum_pool(val, self.scale)
            else:
                self.loss_in_HR = True
        elif self.model == 'simpleHRa':
            feat_l_up = self.up_(self.feat_l, 8)

            # Estimated sub-pixel features from LR
            feat = tf.concat([self.feat_h, feat_l_up[..., 3:]], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
            if not self.is_hr_label:
                for key, val in y_hat.iteritems():
                    val_ = sum_pool(val, self.scale)
                    y_hat[key] = tf.layers.conv2d(val_, 128, 1, activation=None, padding='same')
            else:
                self.loss_in_HR = True
        elif self.model == 'simpleHR1':

            HR_hat_down = self.down_(self.feat_l, 3)

            feat = tf.concat([HR_hat_down, self.feat_l[..., 3:]], axis=3)

            y_hat = simple(feat, n_channels=1, is_training=self.is_training)
        else:
            print('Model {} not defined'.format(self.model))
            sys.exit(1)
        if self.args.is_out_relu:
            y_hat['reg'] = tf.nn.relu(y_hat['reg'])
        return y_hat, HR_hat

    def compute_loss(self, y_hat,HR_hat):

        int_ = lambda x: tf.cast(x, dtype=tf.int64)

        if self.is_hr_label and not self.loss_in_HR:
            #TODO check what happens with labels == -1
            self.labels = sum_pool(self.labels, self.scale, name='Label_down')

        self.label_sem = tf.squeeze(int_(tf.greater(self.labels, self.sem_threshold)), axis=3)

        self.w = self.float_(tf.where(tf.greater_equal(self.labels, 0.0),
                            tf.ones_like(self.labels),  ## for wherever i have labels
                            tf.zeros_like(self.labels)))

        loss_reg = tf.losses.mean_squared_error(labels=self.labels, predictions=y_hat['reg'], weights=self.w)

        loss_sem = tf.losses.sparse_softmax_cross_entropy(labels=self.label_sem, logits=y_hat['sem'], weights=self.w)

        W = [v for v in tf.trainable_variables() if ('weights' in v.name or 'kernel' in v.name)]

        # Lambda_weights is always rescaled with 0.0005
        l2_weights = tf.add_n([0.0005 * tf.nn.l2_loss(v) for v in W])

        if HR_hat is not None:
            loss_sr = tf.nn.l2_loss(HR_hat - self.feat_h)
        else:
            loss_sr = 0
        self.loss123 = self.args.lambda_reg * loss_reg + (1.0 - self.args.lambda_reg) * loss_sem + self.args.lambda_sr * loss_sr
        self.loss_w = self.args.lambda_weights * l2_weights
        self.loss = self.loss123 + self.loss_w

        grads = tf.gradients(self.loss, W, name='gradients')
        norm = tf.add_n([tf.norm(g, name='norm') for g in grads])

        tf.summary.scalar('loss/reg', loss_reg)
        tf.summary.scalar('loss/sem', loss_sem)
        if HR_hat is not None: tf.summary.scalar('loss/SR', loss_sr)
        tf.summary.scalar('loss/L2Weigths', l2_weights)
        tf.summary.scalar('loss/L2Grad', norm)

    def compute_summaries(self,y_hat,HR_hat):
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
        if self.loss_in_HR:
            f1 = lambda x: tf.where(x ==-1,x,x*(2.0/tf.reduce_max(x)))
            inv_reg_ = lambda x: uint8_(colorize(f1(x), vmin=-1, vmax=2.0, cmap='hot'))
        else:
            inv_reg_ = lambda x: uint8_(colorize(x, vmin=-1,vmax=2.0, cmap='hot'))
        inv_sem_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=1, cmap='hot'))
        int_ = lambda x: tf.cast(x, dtype=tf.int64)

        pred_class = tf.argmax(y_hat['sem'], axis=3)
        labels = self.labels
        label_sem = self.label_sem
        w = self.w
        y_hat_reg = y_hat['reg']

        feat_l_up = tf.map_fn(inv_, bilinear(self.feat_l, size=self.patch_size * self.scale), dtype=tf.uint8)

        feat_l_ = tf.map_fn(inv_, self.feat_l, dtype=tf.uint8)

        feat_h_down = uint8_(
                bilinear(self.feat_h, self.patch_size, name='HR_down')) if self.feat_h is not None else feat_l_

        image_array_top = tf.concat(axis=2, values=[tf.map_fn(inv_reg_, labels, dtype=tf.uint8),
                                                    tf.map_fn(inv_reg_, y_hat_reg, dtype=tf.uint8)])

        image_array_mid = tf.concat(axis=2, values=[tf.map_fn(inv_sem_, label_sem, dtype=tf.uint8),
                                                    tf.map_fn(inv_sem_, int_(pred_class), dtype=tf.uint8)])

        if self.loss_in_HR:
            assert (self.labels.shape[1:3] == self.feat_h.shape[1:3])

            image_array_bottom = tf.concat(axis=2, values=[feat_l_up, uint8_(self.feat_h)])
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.summary.image('HR_Loss/HR',
                             image_array,
                             max_outputs=2)

            # Compute summaries in LR space

            labels = sum_pool(self.labels, self.scale, name='Label_down')
            y_hat_reg = sum_pool(y_hat['reg'], self.scale, name='y_reg_down')

            label_sem = tf.squeeze(int_(tf.greater(labels, self.sem_threshold)), axis=3)
            pred_class= tf.squeeze(tf.round(avg_pool(self.float_(pred_class),self.scale,name='sem_down')),axis=3)

            w = self.float_(tf.where(tf.greater_equal(labels, 0.0),
                                     tf.ones_like(labels),  ## for wherever i have labels
                                     tf.zeros_like(labels)))
            inv_reg_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=2.0, cmap='hot'))

            image_array_top = tf.concat(axis=2, values=[tf.map_fn(inv_reg_, labels, dtype=tf.uint8),
                                                        tf.map_fn(inv_reg_, y_hat_reg, dtype=tf.uint8)])
            image_array_mid = tf.concat(axis=2, values=[tf.map_fn(inv_sem_, label_sem, dtype=tf.uint8),
                                                        tf.map_fn(inv_sem_, int_(pred_class), dtype=tf.uint8)])
            image_array_bottom = tf.concat(axis=2, values=[feat_l_, feat_h_down])
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.summary.image('HR_Loss/LR',
                             image_array,
                             max_outputs=2)

        else:

            image_array_bottom = tf.concat(axis=2, values=[feat_l_, feat_h_down])
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.summary.image('LR_Loss/LR',
                             image_array,
                             max_outputs=2)

        if HR_hat is not None:
            image_array = tf.concat(axis=2,
                                    values=[feat_l_up, uint8_(HR_hat), uint8_(self.feat_h)])

            tf.summary.image('HR_hat-HR',
                             image_array,
                             max_outputs=2)

        args_tensor = tf.make_tensor_proto([([k, str(v)]) for k, v in sorted(self.args.__dict__.items())])
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.Summary()
        summary.value.add(tag="FLAGS", metadata=meta, tensor=args_tensor)
        summary_writer = tf.summary.FileWriter(self.args.model_dir)
        summary_writer.add_summary(summary)

        if not self.is_training:
            # Compute evaluation metrics.
            self.eval_metric_ops = {
                'metrics/mae': tf.metrics.mean_absolute_error(
                    labels=labels, predictions=y_hat_reg, weights=w),
                'metrics/mse': tf.metrics.mean_squared_error(
                    labels=labels, predictions=y_hat_reg, weights=w),
                'metrics/iou': tf.metrics.mean_iou(labels=label_sem, predictions=pred_class, num_classes=2,
                                                   weights=w),
                'metrics/prec': tf.metrics.precision(labels=label_sem, predictions=pred_class, weights=w),
                'metrics/acc': tf.metrics.accuracy(labels=label_sem, predictions=pred_class, weights=w),
                'metrics/recall': tf.metrics.recall(labels=label_sem, predictions=pred_class, weights=w)}

            if HR_hat is not None:
                self.eval_metric_ops['metrics/semi_loss'] = tf.metrics.mean_squared_error(HR_hat, self.feat_h)
                self.eval_metric_ops['metrics/s2nr'] = snr_metric(HR_hat, self.feat_h)
        else:
            self.eval_metric_ops = None

    def compute_train_op(self):

        if self.args.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)

        else:
            print('option not defined')
            sys.exit(1)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            step = tf.train.get_global_step()
            if self.args.l2_weights_every is None:
                self.train_op = optimizer.minimize(self.loss, global_step=step)
                if self.args.optimizer == 'adam':
                    lr_adam = get_lr_ADAM(optimizer, learning_rate=0.01)
                    tf.summary.scalar('loss/adam_lr', lr_adam)
            else:
                train_op1 = optimizer.minimize(self.loss123, global_step=step)
                train_op2 = optimizer.minimize(self.loss_w, global_step=step)
                self.train_op = tf.cond(tf.equal(0, tf.to_int32(tf.mod(step, self.args.l2_weights_every))),
                                   true_fn=lambda: train_op1,
                                   false_fn=lambda: train_op2)

