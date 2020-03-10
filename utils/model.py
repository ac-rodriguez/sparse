import tensorflow as tf

import numpy as np
cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

# from AdaBound import AdaBoundOptimizer

from utils.colorize import colorize, inv_preprocess_tf
from utils.models_reg import simple, countception

import utils.tools_tf as tools
import utils.models_semi as semi

class Model:
    def __init__(self, params):
        self.args = params['args']
        self.model_dir = params['model_dir']

        self.scale = 1
        self.is_slim = self.args.is_slim_eval

        self.float_ = lambda x: tf.cast(x, dtype=tf.float32)

        self.sem_threshold = 0
        self.max_output_img = 1
        self.model = self.args.model
        self.two_ds = True
        self.n_classes = 1
        if 'palmcoco' in self.args.dataset:
            self.n_classes = 2
        elif hasattr(self.args,'save_dir'):
            if 'palmcoco' in self.args.model_dir:
                self.n_classes = 2

        self.add_yhat = True

        self.is_first_train = True # added for init variables only in the first train loop
        self.pad = self.args.sq_kernel*16//2 if self.args.sq_kernel else 16

    def get_w(self, lab, is_99=False):

        w = tf.greater_equal(lab,0.0)
        if is_99:
            w = tf.math.logical_and(w, tf.less(lab, 99.0))  # added for cases where we have sem label but no density codes as 99
        return self.float_(w)

    def get_sem(self,lab, return_w = False):
        int_ = lambda x: tf.cast(x, dtype=tf.int32)
        w = self.get_w(lab)

        zeros_background = tf.zeros_like(lab[...,-1])[...,tf.newaxis]
        lab = tf.concat((zeros_background,lab),axis=-1) # adding background class for argmax
        label_sem = tf.argmax(int_(tf.greater(lab, self.sem_threshold)), axis=3, output_type=tf.int32)

        if return_w:
            return label_sem,w
        else:
            return label_sem

    def model_fn(self, features, labels, mode):
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not self.is_training and self.is_slim: self.max_output_img = 2

        if isinstance(features, dict):
            self.feat_l = features['feat_l']

            if self.two_ds:
                self.feat_lU = features['feat_lU']
        else:
            self.feat_l = features
            self.feat_h = None

        self.patch_size = self.feat_l.shape[1]

        self.config = None

        self.compute_predicitons()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=self.y_hat)

        # self.features = features
        if self.args.sq_kernel is None:
            self.sem_threshold = 1e-5


        self.labels = self.compute_labels_ls(labels, self.scale)
        self.labelsh = labels

        self.losses = []
        self.scale_losses = []
        self.compute_loss()
        self.compute_summaries()

        iters_epoch = self.args.train_patches // self.args.batch_size
        epochs = tf.compat.v1.train.get_global_step() / iters_epoch
        if mode == tf.estimator.ModeKeys.TRAIN:
            logging_hook = tf.estimator.LoggingTensorHook({"EPOCH": epochs}, every_n_iter=iters_epoch)
            self.compute_train_op()
            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op, training_hooks=[logging_hook])
        tf.compat.v1.summary.scalar('global_step/epoch',epochs)
        # Add summary hook for image summary
        eval_summary_hook = tf.estimator.SummarySaverHook(
            save_steps=200,
            output_dir=self.model_dir + '/eval',
            summary_op=tf.compat.v1.summary.merge_all())

        return tf.estimator.EstimatorSpec(
            mode, loss=self.loss, eval_metric_ops=self.eval_metric_ops, evaluation_hooks=[eval_summary_hook])

    def compute_predicitons(self):
        size = self.patch_size * self.args.scale
        self.is_small = not ('1' in self.model)
        # Baseline Models
        if self.model == 'simple' or self.model == 'simpleA':  # Baseline  No High Res for training
            earlyl = self.feat_l
            if self.model == 'simpleA':
                earlyl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat, mid, latel = simple(earlyl, n_classes=self.n_classes, is_training=self.is_training,
                                            return_feat=True)

        elif 'simpleA' in self.model:

            depth = int(self.model.replace('simpleA',''))
            earlyl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat, mid, latel = simple(earlyl, n_classes=self.n_classes, is_training=self.is_training,
                                            return_feat=True, deeper=depth)

        elif self.model == 'count':
            self.y_hat = countception(self.feat_l,pad=self.pad, is_training=self.is_training, config_volume=self.config)

        elif self.model == 'countA':

            earlyl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat,midl,latel = countception(earlyl, pad=self.pad, is_training=self.is_training,config_volume=self.config, is_return_feat=True)

        else:
            raise ValueError('Model {} not defined'.format(self.model))
        if self.args.is_out_relu:
            self.y_hat['reg'] = tf.nn.relu(self.y_hat['reg'])

    def compute_label_sem(self):
        pass

    def compute_loss(self):
        labels = self.labels
        label_sem, w = self.get_sem(labels, return_w=True)

        if self.add_yhat:
            lam_evol = tools.evolving_lambda(self.args, height=self.args.low_task_evol) if self.args.low_task_evol is not None else 1.0
            if self.args.lambda_reg > 0.0:
                w_reg = self.get_w(labels,is_99=True)
                loss_reg = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=self.y_hat['reg'], weights=w_reg)
                self.losses.append(loss_reg)
                self.scale_losses.append(self.args.lambda_reg * lam_evol)
                # self.lossTasks+= self.args.lambda_reg * loss_reg * lam_evol
                tf.compat.v1.summary.scalar('loss/reg', loss_reg)
                if self.args.combinatorial_loss is not None:
                    for i in range(self.args.combinatorial_loss):
                        y_ = tools.sum_pool(self.y_hat['reg'], (i + 1) * 2)
                        gt_ = self.compute_labels_ls(self.labels, (i + 1) * 2)
                        w_ = self.get_w(gt_, is_99=True)
                        loss_ = tf.compat.v1.losses.mean_squared_error(labels=gt_, predictions=y_, weights=w_)
                        self.losses.append(loss_)
                        self.scale_losses.append(self.args.lambda_reg / self.args.combinatorial_loss)
                    norm_cross_corr = tools.pair_distance(self.y_hat['reg'],self.labels)
                    self.losses.append(tf.reduce_sum(input_tensor=tf.square(tf.maximum(0., 1.0 - norm_cross_corr))))
                    self.scale_losses.append(self.args.lambda_reg *1)

            if self.args.lambda_reg < 1.0:
                w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
                loss_sem = cross_entropy(labels=label_sem, logits=self.y_hat['sem'], weights=w_)
                self.losses.append(loss_sem)
                self.scale_losses.append((1.0 - self.args.lambda_reg) * lam_evol)
                # self.lossTasks+= (1.0 - self.args.lambda_reg) * loss_sem * lam_evol
                tf.compat.v1.summary.scalar('loss/sem', loss_sem)

        # L2 weight Regularizer
        W = [v for v in tf.compat.v1.trainable_variables() if not 'teacher' in v.name] # if ('weights' in v.name or 'kernel' in v.name)
        # Lambda_weights is always rescaled with 0.0005
        l2_weights = tf.add_n([tf.nn.l2_loss(v) for v in W], name='w_loss')
        tf.compat.v1.summary.scalar('loss/L2Weigths', l2_weights)
        # self.loss_w = self.args.lambda_weights * l2_weights
        self.losses.append(l2_weights)
        self.scale_losses.append(self.args.lambda_weights)



        loss_with_scales = [a * b for a, b in zip(self.losses, self.scale_losses)]
        self.loss = tf.reduce_sum(input_tensor=loss_with_scales)
        # self.loss = self.lossTasks + self.loss_w
        grads = tf.gradients(ys=self.loss, xs=W, name='gradients')
        norm = tf.add_n([tf.norm(tensor=g, name='norm') for g in grads])

        tf.compat.v1.summary.scalar('loss/L2Grad', norm)

    def compute_summaries(self):
        graph = tf.compat.v1.get_default_graph()
        self.eval_metric_ops = {}
        try:
            mean_train = graph.get_tensor_by_name("mean_train_k:0")
            scale = graph.get_tensor_by_name("std_train_k:0")
            max_dens = graph.get_tensor_by_name("max_dens_k:0")
        except KeyError:
            # if constants are not defined in the graph yet,
            # after loading pre-trained networks tf.Variables are used with the correct values
            mean_train = tf.Variable(np.zeros(11), name='mean_train', trainable=False, dtype=tf.float32)
            scale = tf.Variable(np.ones(11), name='std_train', trainable=False, dtype=tf.float32)
            max_dens = tf.Variable(np.ones(1), name='max_dens', trainable=False, dtype=tf.float32)

        uint8_ = lambda x: tf.cast(x * 255.0, dtype=tf.uint8)
        inv_ = lambda x: inv_preprocess_tf(x, mean_train, scale_luminosity=scale, s2=True)
        max_ = 20. if self.args.dataset == 'palmage' else 2.0
        f1 = lambda x: tf.compat.v1.where(x == -1, x, x * (2.0 / max_dens))
        inv_regh_ = lambda x: uint8_(colorize(f1(x), vmin=-1, vmax=max_, cmap='viridis'))
        inv_reg_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=max_, cmap='viridis'))
        inv_sem_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=self.n_classes, cmap='jet'))
        inv_difreg_ = lambda x: uint8_(colorize(x,vmin=-2,vmax=2, cmap='coolwarm'))


        y_hat_reg = self.y_hat['reg']
        pred_class = tf.argmax(input=self.y_hat['sem'], axis=3)

        y_hat_reg_down = y_hat_reg
        pred_class_down = pred_class

        feat_l_ = tf.map_fn(inv_, self.feat_l, dtype=tf.uint8)

        feat_h_down = feat_l_

        # compute summaries in LR space

        label_sem, w = self.get_sem(self.labels, return_w=True)

        image_array_top = self.concat_reg(self.labels, y_hat_reg_down, inv_reg_, inv_difreg_)
        image_array_mid = self.concat_sem(label_sem, pred_class_down, inv_sem_, inv_difreg_)

        image_array_bottom = tf.concat(axis=2, values=[feat_l_, feat_h_down, tf.zeros_like(feat_h_down)])

        if self.args.lambda_reg == 1.0:
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_bottom])
        elif self.args.lambda_reg == 0.0:
            image_array = tf.concat(axis=1, values=[image_array_mid, image_array_bottom])
        else:
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

        tf.compat.v1.summary.image('LR', image_array, max_outputs=self.max_output_img)

        args_tensor = tf.compat.v1.make_tensor_proto([([k, str(v)]) for k, v in sorted(self.args.__dict__.items())])
        meta = tf.compat.v1.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.compat.v1.Summary()
        summary.value.add(tag="FLAGS", metadata=meta, tensor=args_tensor)
        summary_writer = tf.compat.v1.summary.FileWriter(self.args.model_dir)
        summary_writer.add_summary(summary)

        if not self.is_training:
            # Compute evaluation metrics.
            labels = self.labels

            metrics_reg = {
                'metrics/mae': tf.compat.v1.metrics.mean_absolute_error(labels=labels, predictions=y_hat_reg_down, weights=w),
                'metrics/mse': tf.compat.v1.metrics.mean_squared_error(labels=labels, predictions=y_hat_reg_down, weights=w),
                }
            w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
            metrics_sem = {
                'metrics/iou': tf.compat.v1.metrics.mean_iou(labels=label_sem, predictions=pred_class_down, num_classes=self.n_classes+1, weights=w_),
                'metrics/prec': tf.compat.v1.metrics.precision(labels=label_sem, predictions=pred_class_down, weights=w_),
                'metrics/acc': tf.compat.v1.metrics.accuracy(labels=label_sem, predictions=pred_class_down, weights=w_),
                'metrics/recall': tf.compat.v1.metrics.recall(labels=label_sem, predictions=pred_class_down, weights=w_)}

            if self.args.lambda_reg > 0.0:
                self.eval_metric_ops.update(metrics_reg)
            if self.args.lambda_reg < 1.0:
                self.eval_metric_ops.update(metrics_sem)
        else:
            self.eval_metric_ops = None

    def get_reg(self,labels):
        return labels

    def concat_reg(self, labels, y_hat_reg, inv_reg_, inv_difreg_):
        labels_all = self.get_reg(labels)
        img_out = []
        for i in range(self.n_classes):
            labels = tf.expand_dims(labels_all[..., i], axis=-1)
            y_hat_reg_ = tf.expand_dims(y_hat_reg[..., i], axis=-1)
            image_array_top = tf.map_fn(inv_reg_, tf.concat(axis=2, values=[labels, y_hat_reg_, ]), dtype=tf.uint8)
            image_array_top = tf.concat(axis=2, values=[image_array_top,
                                                        tf.map_fn(inv_difreg_, labels - y_hat_reg_, dtype=tf.uint8)])
            img_out.append(image_array_top)

        return tf.concat(values=img_out,axis=1)
    @staticmethod
    def concat_sem(label_sem, pred_class, inv_sem_, inv_difreg_):
        int_ = lambda x: tf.cast(x, dtype=tf.int32)

        image_array_mid = tf.map_fn(inv_sem_, tf.concat(axis=2, values=[label_sem, int_(pred_class)]), dtype=tf.uint8)
        image_array_mid = tf.concat(axis=2, values=[image_array_mid,
                                                    tf.map_fn(inv_difreg_, label_sem - int_(pred_class),
                                                              dtype=tf.uint8)])
        return image_array_mid

    def compute_labels_ls(self,labels,scale):
        x = tf.clip_by_value(labels,0,1000)
        x = tools.sum_pool(x, scale)

        xm = tools.max_pool(labels, scale)
        x = tf.compat.v1.where(tf.equal(xm,-1),xm,x, name='Label_down')
        return x

    def get_optimiter(self):
        if self.args.optimizer == 'adagrad':
            optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == 'adam':
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == 'SGD':
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == 'momentum':
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.args.lr, momentum=0.9, use_nesterov=True)
        elif self.args.optimizer == 'annealing':
            learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.001, power=0.75)
            tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError('optimizer {} not defined'.format(self.args.optimizer))
        return optimizer
    def compute_train_op(self):

        optimizer = self.get_optimiter()
        if self.is_first_train:
            tools.analyze_model()
            self.is_first_train = False
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        vars_train = [x for x in tf.compat.v1.trainable_variables() if not 'teacher' in x.name]
        with tf.control_dependencies(update_ops):
            step = tf.compat.v1.train.get_global_step()
            if self.args.l2_weights_every is None:
                self.train_op = optimizer.minimize(self.loss, global_step=step, var_list=vars_train)

                if self.args.optimizer == 'adam':
                    lr_adam = tools.get_lr_ADAM(optimizer, learning_rate=self.args.lr)
                    tf.compat.v1.summary.scalar('loss/adam_lr', tf.math.log(lr_adam))
            else:
                raise NotImplementedError

