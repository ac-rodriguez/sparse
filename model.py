import tensorflow as tf
# from tensorflow.contrib.tensorboard.plugins import projector
# import sys
import numpy as np
cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

from AdaBound import AdaBoundOptimizer

from colorize import colorize, inv_preprocess_tf
from models_reg import simple, countception, dl3
import models_sr as sr

import tools_tf as tools
import models_semi as semi
# import min_norm_solvers_tf as solver
# import min_norm_solvers_numpy as solvernp
class Model:
    def __init__(self, params):
        self.args = params['args']
        self.model_dir = params['model_dir']

        self.scale = self.args.scale
        self.is_hr_label = self.args.is_hr_label
        self.is_hr_pred = self.args.is_hr_pred
        self.is_slim = self.args.is_slim_eval
        if self.args.is_fake_hr_label:
            self.is_hr_label = True
        if self.args.is_bilinear:
            self.down_ = lambda x, _: tools.bilinear(x, self.patch_size, name='HR_hat_down')
            self.up_ = lambda x, _: tools.bilinear(x, self.patch_size * self.scale, name='HR_hat_down')
        else:
            self.down_ = lambda x, ch: tf.compat.v1.layers.conv2d(x, ch, 3, strides=self.scale, padding='same')
            self.up_ = lambda x, ch: tf.compat.v1.layers.conv2d_transpose(x, ch, 3, strides=self.scale, padding='same')

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

        self.hr_emb = False
        self.add_yhath = False
        self.add_yhat = True

        self.is_domain_transfer = False
        self.is_semi = self.args.semi is not None
        self.is_sr = False
        self.sr_on_labeled = True
        self.is_adversarial = False
        self.not_s2 = "vaihingen" in self.args.dataset
        self.is_first_train = True # added for init variables only in the first train loop
        self.pad = self.args.sq_kernel*16//2 if self.args.sq_kernel else 16
        if ('HR' in self.model or 'SR' in self.model or 'DA_h' in self.model or 'B_h' in self.model) and not '_l' in self.model and self.is_hr_label:
            self.hr_emb = True

    def get_w(self, lab, is_99=False):
        if self.not_s2:
            lab = tf.expand_dims(lab[...,0],-1)
        w = tf.greater_equal(lab,0.0)
        if is_99:
            w = tf.math.logical_and(w, tf.less(lab, 99.0))  # added for cases where we have sem label but no density codes as 99
        return self.float_(w)

    def get_sem(self,lab, return_w = False):
        int_ = lambda x: tf.cast(x, dtype=tf.int32)
        w = self.get_w(lab)

        if self.not_s2:
            label_sem = int_(tf.equal(lab[...,0], 5.0))
        else:

            zeros_background = tf.zeros_like(lab[...,-1])[...,tf.newaxis]
            lab = tf.concat((zeros_background,lab),axis=-1) # adding background class for argmax
            label_sem = tf.argmax(int_(tf.greater(lab, self.sem_threshold)), axis=3, output_type=tf.int32)
            w_ = tf.reduce_any(tf.greater(w, 0),-1)
            # label_sem = tf.compat.v1.where(w_, label_sem,
            #                      tf.ones_like(label_sem, dtype=tf.int32) * -1)
        if return_w:
            return label_sem,w
        else:
            return label_sem

    def model_fn(self, features, labels, mode):
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not self.is_training and self.is_slim: self.max_output_img = 2

        if isinstance(features, dict):
            self.feat_l = features['feat_l']
            self.feat_h = features['feat_h'] if 'feat_h' in features.keys() else None
            if self.two_ds:
                self.feat_lU = features['feat_lU']
                self.feat_hU = features['feat_hU'] if 'feat_hU' in features.keys() else None
        else:
            self.feat_l = features
            self.feat_h = None

        self.patch_size = self.feat_l.shape[1]

        if self.args.is_same_volume:
            self.config = {'hr':self.is_hr_label,'scale':self.scale,'patch':self.patch_size, 'last':'last' in self.model}
        else:
            self.config = None

        if self.args.degraded_hr and self.is_training:

            self.feat_h = tools.low_pass_filter(self.feat_h, self.args, blur_probability=1.0)
        # else:
        #     self.feat_h = tools.low_pass_filter(self.feat_h, self.args, blur_probability=1.0, progressive=False)


        self.compute_predicitons()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=self.y_hat)

        # self.features = features
        if self.args.sq_kernel is None:
            self.sem_threshold = 1e-5

        if self.is_hr_label:
            if self.args.is_fake_hr_label:
                self.labels = labels
                self.labelsh = tf.image.resize_nearest_neighbor(self.labels, size=(
                self.patch_size * self.scale, self.patch_size * self.scale)) / (self.scale ** 2)
            else:
                self.labels = self.compute_labels_ls(labels, self.scale)
                self.labelsh = labels
        else:
            self.labels = labels
            self.labelsh = None

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
            if self.is_semi and self.is_training:
                earlylU = self.feat_lU
                if self.model == 'simpleA':
                    earlylU = semi.encode_same(self.feat_lU, is_training=self.is_training, is_bn=True, is_small=self.is_small)
                self.y_hatU,midlU,latelU = simple(earlylU, n_classes=self.n_classes, is_training=self.is_training,
                                                  return_feat=True)
                self.Zl = latel
                self.ZlU =latelU
        elif 'simpleA' in self.model:

            depth = int(self.model.replace('simpleA',''))
            earlyl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat, mid, latel = simple(earlyl, n_classes=self.n_classes, is_training=self.is_training,
                                            return_feat=True, deeper=depth)
            if self.is_semi and self.is_training:
                earlylU = semi.encode_same(self.feat_lU, is_training=self.is_training, is_bn=True, is_small=self.is_small)
                self.y_hatU,midlU,latelU = simple(earlylU, n_classes=self.n_classes, is_training=self.is_training,
                                                  return_feat=True, deeper=depth)
                self.Zl = latel
                self.ZlU =latelU

        elif self.model == 'count' or self.model == 'counth':
            self.y_hat = countception(self.feat_l,pad=self.pad, is_training=self.is_training, config_volume=self.config)
            # if self.two_ds:
            #         _ = countception(self.feat_lU,pad=self.pad, is_training=self.is_training, is_return_feat=True,  config_volume=self.config)

            if self.model == 'counth':  # using down-sampled HR images too
                self.add_yhath = True
                feat_h_down = self.down_(self.feat_h,3)
                feat_h_down = tf.compat.v1.layers.conv2d(feat_h_down,self.feat_l.shape[-1],kernel_size=1,strides=1)
                self.y_hath = countception(feat_h_down, pad=self.pad, is_training=self.is_training)
        elif self.model == 'countA' or self.model == 'countAh':

            earlyl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat,midl,latel = countception(earlyl, pad=self.pad, is_training=self.is_training,config_volume=self.config, is_return_feat=True)

            if self.model == 'countAh':  # using down-sampled HR images too
                self.add_yhath = True
                feat_h_down = self.down_(self.feat_h,3)
                feat_h_down = tf.compat.v1.layers.conv2d(feat_h_down,earlyl.shape[-1],kernel_size=1,strides=1)
                self.y_hath = countception(feat_h_down, pad=self.pad, is_training=self.is_training)
            if self.is_semi:
                earlylU = semi.encode_same(self.feat_lU, is_training=self.is_training, is_bn=True, is_small=self.is_small)
                self.y_hatU,midlU,latelU = countception(earlylU, pad=self.pad, is_training=self.is_training, config_volume=self.config,is_return_feat=True)
                self.Zl = latel
                self.ZlU =latelU

        elif self.model == 'dl3':
            self.y_hat = dl3(inputs=self.feat_l, n_channels=2, is_training=self.is_training)

        elif self.model == 'dl3B_h':
            feat_l_up = self.up_(self.feat_l, 8)

            self.y_hat = dl3(inputs=feat_l_up, n_channels=2, is_training=self.is_training)

        elif self.model == 'dl3HR_l':
            Ench = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            self.y_hat = dl3(inputs=Ench, n_channels=2, is_training=self.is_training)

        elif self.model == 'dl3HR_h':
            self.y_hat = dl3(inputs=self.feat_h, n_channels=2, is_training=self.is_training)

        elif self.model == 'countSR' or self.model == 'countSRu' or self.model =='countSRonly':
            self.is_sr = True
            self.HR_hat = sr.SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)
            self.HR_hatU = sr.SR_task(feat_l=self.feat_lU, size=size, is_batch_norm=True, is_training=self.is_training)

            feat_l_up = self.up_(self.feat_l, 8)
            feat = tf.concat([self.HR_hat, feat_l_up], axis=3)

            self.y_hat = countception(feat,pad=self.pad, is_training=self.is_training,  config_volume=self.config)
            if self.model == 'countSRu':
                self.sr_on_labeled = False
            if not self.is_hr_label: # added as a baseline
                for key, val in self.y_hat.iteritems():
                    self.y_hat[key] = tools.bilinear(val, self.patch_size)
            else:
                assert self.hr_emb
            if 'only' in self.model:
                assert self.args.lambda_sr > 0, 'set a lambda_sr > 0'
                self.add_yhat = False

        elif self.model == 'countSRs' or self.model == 'countSRsonly':
            self.is_sr = True
            self.HR_hat = semi.decode(self.feat_l, scale=self.scale, is_bn=True, is_training=self.is_training, n_feat_last=3)
            self.HR_hatU = semi.decode(self.feat_lU, scale=self.scale, is_bn=True, is_training=self.is_training, n_feat_last=3)

            feat_l_up = self.up_(self.feat_l, 8)
            feat = tf.concat([self.HR_hat, feat_l_up], axis=3)

            self.y_hat = countception(feat,pad=self.pad, is_training=self.is_training, config_volume=self.config)
            if not self.is_hr_label: # added as a baseline
                for key, val in self.y_hat.iteritems():
                    self.y_hat[key] = tools.bilinear(val, self.patch_size)
            else:
                assert self.hr_emb
            if 'only' in self.model:
                assert self.args.lambda_sr > 0, 'set a lambda_sr > 0'
                self.add_yhat = False
        elif self.model == 'countSRA' or self.model == 'countSRAonly':
            self.is_sr = True
            self.HR_hat = semi.decode(self.feat_l, scale=self.scale, is_bn=True, is_training=self.is_training, n_feat_last=3)
            self.y_hat = countception(self.HR_hat,pad=self.pad, is_training=self.is_training, config_volume=self.config)
            assert self.hr_emb
            if 'only' in self.model:
                assert self.args.lambda_sr > 0, 'set a lambda_sr > 0'
                self.add_yhat = False

        elif self.model == 'countSR_l':
            self.is_sr = True
            self.HR_hat = sr.SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)
            self.HR_hatU = sr.SR_task(feat_l=self.feat_lU, size=size, is_batch_norm=True, is_training=self.is_training)

            feat = semi.encode(self.HR_hat, is_training=self.is_training, is_bn=True, scale=self.args.scale)
            feat = tf.concat([feat, self.feat_l], axis=3)
            self.y_hat = countception(feat,pad=self.pad, is_training=self.is_training)
        elif self.model == 'countHR_lb':
            feat_l_up = self.up_(self.feat_l, 8)
            feat = tf.concat([self.feat_h, feat_l_up[..., 3:]], axis=3)
            self.y_hat , mid,last = countception(feat,pad=self.pad, is_training=self.is_training, is_return_feat=True)
            if self.two_ds:
                feat_lU_up = self.up_(self.feat_lU, 8)
                feat = tf.concat([self.feat_hU, feat_lU_up[..., 3:]], axis=3)
                _, midU, lastU = countception(feat,pad=self.pad, is_training=self.is_training, is_return_feat=True)

            assert self.hr_emb
        elif self.model == 'countHR_la': # Using HR and LR infrared channels

            Ench = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale = self.scale)
            Encl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)
            feat = tf.concat([Ench, Encl], axis=3)

            self.y_hat, self.Zh = countception(feat,pad=self.pad, is_training=self.is_training, is_return_feat=True)
            if self.two_ds:

                EnchU = semi.encode(input=self.feat_hU, is_training=self.is_training, is_bn=True, scale=self.scale)
                EnclU = semi.encode_same(self.feat_lU, is_training=self.is_training, is_bn=True)
                feat = tf.concat([EnchU, EnclU], axis=3)

                _, self.ZhU = countception(feat, pad=self.pad, is_training=self.is_training,
                                                  is_return_feat=True)
        elif self.model == 'countHRA_l': # Using only HR data on lr embedding

            Ench = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale = self.scale)
            self.y_hat = countception(Ench,pad=self.pad, is_training=self.is_training, is_return_feat=False, config_volume=self.config)
            # if self.two_ds:
                # EnchU = semi.encode(input=self.feat_hU, is_training=self.is_training, is_bn=True, scale=self.scale)
                # _, = countception(EnchU, pad=self.pad, is_training=self.is_training, is_return_feat=False, config_volume=self.config)
        elif self.model == 'countHRA_h': # Using only HR data on hr embedding

            Ench = semi.encode_same(input=self.feat_h, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat = countception(Ench,pad=self.pad, is_training=self.is_training, is_return_feat=False, config_volume=self.config)
            # if self.two_ds:
                # EnchU = semi.encode(input=self.feat_hU, is_training=self.is_training, is_bn=True, scale=self.scale)
                # _, self.ZhU = countception(EnchU, pad=self.pad, is_training=self.is_training, is_return_feat=True, config_volume=self.config)

        elif self.model == 'countB_l': # Using only upscaled LR data on lr embedding
            feat_l_up = self.up_(self.feat_l, 8)

            Ench = semi.encode(input=feat_l_up, is_training=self.is_training, is_bn=True, scale = self.scale)
            self.y_hat = countception(Ench,pad=self.pad, is_training=self.is_training, is_return_feat=False, config_volume=self.config)
            # if self.two_ds:
                # EnchU = semi.encode(input=self.feat_hU, is_training=self.is_training, is_bn=True, scale=self.scale)
                # _, = countception(EnchU, pad=self.pad, is_training=self.is_training, is_return_feat=False, config_volume=self.config)
        elif self.model == 'countB_h': # Using only upscaled LR on hr embedding
            feat_l_up = self.up_(self.feat_l, 8)

            Ench = semi.encode_same(input=feat_l_up, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hat = countception(Ench,pad=self.pad, is_training=self.is_training, is_return_feat=False, config_volume=self.config)
            # if self.two_ds:
                # EnchU = semi.encode(input=self.feat_hU, is_training=self.is_training, is_bn=True, scale=self.scale)
                # _, self.ZhU = countception(EnchU, pad=self.pad, is_training=self.is_training, is_return_feat=True, config_volume=self.config)

        elif 'DA_h' in self.model:
            self.daH_models()

        elif 'DA' in self.model:
            self.daL_models()

        else:
            raise ValueError('Model {} not defined'.format(self.model))
        if self.args.is_out_relu:
            self.y_hat['reg'] = tf.nn.relu(self.y_hat['reg'])
        if self.is_domain_transfer and not self.is_slim:
            Emb = tf.concat((self.Zh, self.Zl), axis=0, name='Embeddings')
            Label = tf.concat((tf.ones(tf.shape(input=self.Zh)[:-1]), tf.zeros(tf.shape(input=self.Zh)[:-1])), axis=0,
                              name='EmbeddingsLabel')

    def compute_label_sem(self):
        pass

    def compute_loss(self):
        labels = self.labelsh if self.hr_emb else self.labels
        label_sem, w = self.get_sem(labels, return_w=True)
        if "vaihingen" in self.args.dataset:
            labels = tf.expand_dims(labels[...,0],-1)
        # self.lossTasks = 0.0
        if self.is_training and self.args.distill_from is not None:
            if self.args.is_distill_only:
                self.add_yhat = False
        if self.add_yhat:
            # lam_evol = tools.evolving_lambda(self.args)
            # lam_evol = 1.0
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

        if self.add_yhath and (self.is_training or not self.is_slim):

            lam_evol = tools.evolving_lambda(self.args, height=self.args.high_task_evol) if self.args.high_task_evol is not None else 1.0
            if self.args.lambda_reg > 0.0:
                w_reg = self.get_w(labels,is_99=True)
                loss_reg = tf.compat.v1.losses.mean_squared_error(labels=labels, predictions=self.y_hath['reg'], weights=w_reg)
                self.losses.append(loss_reg)
                self.scale_losses.append(self.args.lambda_reg * lam_evol)
                # self.lossTasks += self.args.lambda_reg * loss_reg * lam_evol
                tf.compat.v1.summary.scalar('loss/regH', loss_reg)
            if self.args.lambda_reg < 1.0:
                w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
                loss_sem = cross_entropy(labels=label_sem, logits=self.y_hath['sem'], weights=w_)
                self.losses.append(loss_sem)
                self.scale_losses.append((1.0 - self.args.lambda_reg) * lam_evol)
                # self.lossTasks+= (1.0 - self.args.lambda_reg) * loss_sem * lam_evol
                tf.compat.v1.summary.scalar('loss/semH', loss_sem)
        # SR loss
        if self.is_sr and self.args.lambda_sr > 0:
            if self.args.sr_after is not None:
                if self.args.sr_after > 0:
                    w1 = tf.compat.v1.where(tf.greater(self.args.sr_after, tf.compat.v1.train.get_global_step()), 0., 1.)
                else:
                    w1 = tf.compat.v1.where(tf.greater(self.args.sr_after, tf.compat.v1.train.get_global_step()), 1., 0.)
            else:
                w1 = 1.
            if self.sr_on_labeled:
                loss_sr = tf.nn.l2_loss(self.HR_hat - self.feat_h)
            else:
                loss_sr = tf.nn.l2_loss(self.HR_hatU - self.feat_hU)
            tf.compat.v1.summary.scalar('loss/SR', loss_sr)
            self.losses.append(tf.identity(w1*loss_sr,'sr_loss'))
            self.scale_losses.append(self.args.lambda_sr)
            # self.lossTasks += self.args.lambda_sr * w1 * loss_sr

        # if self.args.is_multi_task:
        #
        #     optimizer = self.get_optimiter()
        #
        #     grads = {}
        #     tasks = range(len(self.losses))
        #     shared_vars = [x for x in tf.trainable_variables() if 'countception' in x.name]
        #     for t in tasks:
        #         # only the gradient for each task
        #         grads[t] = [x[0] for x in optimizer.compute_gradients(self.losses[t], var_list=shared_vars) if x[0] is not None]
        #
        #     gn = solver.gradient_normalizers(grads, self.losses, 'loss+')
        #     for t in tasks:
        #         for gr_i in range(len(grads[t])):
        #             grads[t][gr_i] = grads[t][gr_i] / gn[t]
        #     NormSolver = solvernp.MinNormSolverNumpy()
        #     # Frank-Wolfe iteration to compute scales.
        #
        #     sol, min_norm = tf.py_func(NormSolver.find_min_norm_element, [grads[t] for t in tasks], tf.float32)
        #
        #     for i, t in enumerate(tasks):
        #         self.scale_losses[t] = sol[i]

        if self.is_training or not self.is_slim:
            if self.args.distill_from is not None:
                best = True
                if best:
                    distill_from = tools.get_last_best_ckpt(self.args.distill_from,'best/*')
                else:
                    distill_from = self.args.distill_from
                # if self.is_first_train:
                #     scope1 = 'encode_same' if 'h' in self.model else 'encode'
                #     scope2 = 'countception' if 'count' in self.model else 'simple'
                #     dict_vars = [x for x in tf.trainable_variables() if scope1 in x.name or scope2 in x.name]
                #     dict_vars = {x.name.split(':')[0]:x.name.split(':')[0] for x in dict_vars}
                #     tf.train.init_from_checkpoint(distill_from,dict_vars)
                #     print('init {} and {} variables from teacher model'.format(scope1,scope2))
                #
                #     self.is_first_train = False
                self.DIS_models()
                tf.compat.v1.train.init_from_checkpoint(distill_from,{'/': 'teacher/'})
                print('init teacher variables from checkpoint')

                self.add_distilled_loss()
            if self.is_semi:
                self.add_semi_loss()
            if self.args.domain is not None:
                self.add_domain_loss()

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


    def daH_models(self):
        assert self.hr_emb
        self.is_domain_transfer = True
        self.add_yhath = True


        assert 'countDA_h' in self.model, 'model {} not defined for DA'.format(self.model)

        earlyl = semi.decode(self.feat_l, is_training=self.is_training, is_bn=True, scale=self.scale)
        self.y_hat, midl, latel = countception(earlyl, pad=self.pad, is_training=self.is_training, is_return_feat=True,
                                           config_volume=self.config)
        if self.is_training or not self.is_slim:
            earlyh = semi.encode_same(input=self.feat_h, is_training=self.is_training, is_bn=True, is_small=self.is_small)
            self.y_hath, midh, lateh = countception(earlyh, pad=self.pad, is_training=self.is_training, is_return_feat=True,
                                                config_volume=self.config)
        else:
            earlyh = midh = lateh = None

        if 'early' in self.model:
            self.Zl = earlyl
            self.Zh = earlyh
        elif 'mid' in self.model:
            self.Zl = midl
            self.Zh = midh
        elif 'late' in self.model:
            self.Zl = latel
            self.Zh = lateh

        else:
            raise ValueError('Model {} not defined'.format(self.model))

    def daL_models(self):
        assert not self.hr_emb
        self.is_domain_transfer = True
        self.add_yhath = True

        assert 'countDA' in self.model, 'model {} not defined for DA'.format(self.model)
        earlyl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True, is_small=self.is_small)
        self.y_hat, midl, latel = countception(earlyl, pad=self.pad, is_training=self.is_training,
                                               is_return_feat=True,  config_volume=self.config)

        if self.is_training or not self.is_slim:
            earlyh = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            self.y_hath, midh, lateh = countception(earlyh, pad=self.pad, is_training=self.is_training,
                                                    is_return_feat=True,  config_volume=self.config)
        else:
            earlyh = midh = lateh = None

        if 'early' in self.model:
            self.Zl = earlyl
            self.Zh = earlyh
        elif 'mid' in self.model:
            self.Zl = midl
            self.Zh = midh
        elif 'late' in self.model:
            self.Zl = latel
            self.Zh = lateh

        elif self.model == 'countDApair':
            # Train on (HR,y). Test on (Lr,y). LR is paired with HR. Comparison on Higher-level features
            # TODO fix
            encLR = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)
            self.y_hat, self.Zl = countception(encLR, pad=self.pad, is_training=self.is_training,
                                          is_return_feat=True)
            if self.is_training or not self.is_slim:
                encHR = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
                self.y_hath, self.Zh = countception(encHR, pad=self.pad, is_training=self.is_training,
                                              is_return_feat=True)
            if self.is_training:
                self.add_yhat = False
                self.add_yhath = True
            else:
                self.add_yhat = True
                self.add_yhath = False

        elif self.model == 'countDAunpair':
            # Train on (HR,y). Test on (Lr,y). LR is unpaired with HR. Comparison on Higher-level features
            assert not 'L' in self.args.domain, 'unpaired DA should be used with L1 domain loss.'

            encHR = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            encLR = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)
            encLRU = semi.encode_same(self.feat_lU, is_training=self.is_training, is_bn=True)

            _, self.Zl = countception(encLRU, pad=self.pad, is_training=self.is_training,
                                      is_return_feat=True)
            self.y_hath, self.Zh = countception(encHR, pad=self.pad, is_training=self.is_training, is_return_feat=True)
            self.y_hat = countception(encLR, pad=self.pad, is_training=self.is_training)

            if self.is_training:
                self.add_yhat = False
                self.add_yhath = True
            else:
                self.add_yhat = True
                self.add_yhath = False

        else:
            raise ValueError('Model {} not defined'.format(self.model))
    def DIS_models(self):

        with tf.compat.v1.variable_scope('teacher'):
            if 'dl' not in self.model:
                if 'h' in self.model:
                    Ench = semi.encode_same(input=self.feat_h, is_training=self.is_training, is_bn=True, is_small=self.is_small)
                else:
                    Ench = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.args.scale)
                self.y_hat_teacher = countception(Ench, pad=self.pad, is_training=self.is_training, is_return_feat=False,
                                          config_volume=self.config)
            else:
                if 'h' in self.model:
                    Ench = self.feat_h
                else:
                    Ench = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.args.scale)
                self.y_hat_teacher = dl3(inputs=Ench, n_channels=2, is_training=self.is_training)

    def add_semi_loss(self):
        # TODO fix label and labelh separation
        if self.args.semi == 'semiRev':

            if 'HR' in self.model:
                self.score = semi.domain_discriminator(self.Zh, scope_name='domain_semi')
                self.scoreU = semi.domain_discriminator(self.ZhU, scope_name='domain_semi')
            else:
                self.score = semi.domain_discriminator(self.Zl, scope_name='domain_semi')
                self.scoreU = semi.domain_discriminator(self.ZlU, scope_name='domain_semi')

            # Fake predictions towards 0, real preds towards 1
            loss_domain = cross_entropy(labels=tf.zeros_like(self.score[..., 0], dtype=tf.int32), logits=self.score) + \
                          cross_entropy(labels=tf.ones_like(self.score[..., 0], dtype=tf.int32), logits=self.scoreU)

            tf.compat.v1.summary.scalar('loss/domain_semi', loss_domain)
            self.losses.append(loss_domain)
            self.scale_losses.append(1.0)
            # self.loss += loss_domain
        else:
            if self.args.semi == 'semi':
                self.fake_ = semi.discriminator(self.y_hat['reg'])
                self.real_ = semi.discriminator(self.labels)
                # Fake predictions towards 0, real preds towards 1
                self.loss_disc = cross_entropy(labels=tf.zeros_like(self.labels, dtype=tf.int32),logits=self.fake_) + \
                            cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32), logits=self.real_,weights=self.w)

                self.loss_gen = cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32),logits=self.fake_)
            elif self.args.semi == 'semiFeat':
                self.fake_,feat_f = semi.discriminator(self.y_hat['reg'],return_feat=True)
                self.real_,feat_r = semi.discriminator(self.labels, return_feat=True)
                # Fake predictions towards 0, real preds towards 1
                self.loss_disc = cross_entropy(labels=tf.zeros_like(self.labels, dtype=tf.int32),logits=self.fake_) + \
                            cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32), logits=self.real_,weights=self.w)
                self.loss_gen = tf.compat.v1.losses.mean_squared_error(labels=feat_r, predictions=feat_f)

            elif self.args.semi == 'semi1':
                self.fake_ = semi.discriminator(tf.concat((self.y_hat['reg'],self.y_hat['sem']),axis=-1))
                self.real_ = semi.discriminator(tf.concat((self.labels,self.float_(tf.one_hot(self.label_sem,depth=2))),axis=-1))
                # Fake predictions towards 0, real preds towards 1
                self.loss_disc = cross_entropy(labels=tf.zeros_like(self.labels, dtype=tf.int32), logits=self.fake_) + \
                                 cross_entropy(labels=tf.ones_like(self.labels, dtype=tf.int32), logits=self.real_,
                                               weights=self.w)
                self.loss_gen = cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32),logits=self.fake_)

            elif self.args.semi == 'semi1Feat':
                self.fake_, feat_f= semi.discriminator(tf.concat((y_hat['reg'],y_hat['sem']),axis=-1), return_feat=True)
                self.real_, feat_r = semi.discriminator(tf.concat((self.labels,self.float_(tf.one_hot(self.label_sem,depth=2))),axis=-1), return_feat=True)
                # Fake predictions towards 0, real preds towards 1
                self.loss_disc = cross_entropy(labels=tf.zeros_like(self.labels, dtype=tf.int32), logits=self.fake_) + \
                                 cross_entropy(labels=tf.ones_like(self.labels, dtype=tf.int32), logits=self.real_,
                                               weights=self.w)
                self.loss_gen = tf.compat.v1.losses.mean_squared_error(labels=feat_r, predictions=feat_f)

            else:
                raise ValueError('{} model semi-supervised not defined'.format(self.args.semi))
            tf.compat.v1.summary.scalar('loss/gen', self.loss_gen)
            tf.compat.v1.summary.scalar('loss/disc', self.loss_disc)

            # self.loss+= self.loss_gen
            self.losses.append(self.loss_gen)
            lambda_semi = tools.evolving_lambda(self.args, height=1.0)
            tf.compat.v1.summary.scalar('loss/lambda_semi', lambda_semi)
            self.scale_losses.append(lambda_semi)
    def add_domain_loss(self):
        assert self.is_domain_transfer, ' domain-loss not defined for model:{} '.format(self.model)
        lambda_domain = tools.evolving_lambda(self.args, height=1.0)
        tf.compat.v1.summary.scalar('loss/lambda_domain', lambda_domain)

        if self.args.domain == 'DANNlarge' or self.args.domain == 'DANNlargeL':

            self.scoreh = semi.domain_discriminator(self.Zh)
            self.scorel = semi.domain_discriminator(self.Zl)

            # Fake predictions towards 0, real preds towards 1
            loss_domain = cross_entropy(labels=tf.zeros_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scoreh) + \
                             cross_entropy(labels=tf.ones_like(self.scorel[...,0], dtype=tf.int32), logits=self.scorel)
            if 'L' in self.args.domain:
                loss_domain+= tf.compat.v1.losses.absolute_difference(self.Zh, self.Zl)

            tf.compat.v1.summary.scalar('loss/domain', loss_domain)
        elif self.args.domain == 'DANN':

            self.scoreh = semi.domain_discriminator_small(self.Zh)
            self.scorel = semi.domain_discriminator_small(self.Zl)
            labels = tf.concat((tf.zeros_like(self.scoreh[...,0], dtype=tf.int32), tf.ones_like(self.scorel[...,0], dtype=tf.int32)), axis=0)
            logits = tf.concat((self.scoreh, self.scorel), axis=0)
            # Fake predictions towards 0, real preds towards 1
            # loss_domain = cross_entropy(labels=tf.zeros_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scoreh) + \
            #                  cross_entropy(labels=tf.ones_like(self.scorel[...,0], dtype=tf.int32), logits=self.scorel)
            loss_domain = cross_entropy(labels,logits)
            tf.compat.v1.summary.scalar('loss/domain', loss_domain)

            tf.compat.v1.summary.histogram('loss/domain_logitsh_0',self.scoreh[...,0])
            tf.compat.v1.summary.histogram('loss/domain_logitsh_1',self.scoreh[...,1])
            tf.compat.v1.summary.histogram('loss/domain_logitsl_0', self.scorel[..., 0])
            tf.compat.v1.summary.histogram('loss/domain_logitsl_1', self.scorel[..., 1])

        elif self.args.domain == 'DANNadv':

            self.scoreh = semi.domain_discriminator_small(self.Zh, is_flip=False)
            self.scorel = semi.domain_discriminator_small(self.Zl, is_flip=False)

            labels = tf.concat(
                (tf.zeros_like(self.scoreh[..., 0], dtype=tf.int32), tf.ones_like(self.scorel[..., 0], dtype=tf.int32)),
                axis=0)
            logits = tf.concat((self.scoreh, self.scorel), axis=0)
            self.loss_disc = lambda_domain*cross_entropy(labels, logits)
            self.vars_d = [v for v in tf.compat.v1.trainable_variables() if 'domain_discriminator' in v.name and not 'teacher' in v.name]
            self.vars_g = [v for v in tf.compat.v1.trainable_variables() if not 'domain_discriminator' in v.name and not 'teacher' in v.name]
            loss_gen = cross_entropy(tf.ones_like(self.scorel[..., 0], dtype=tf.int32),self.scorel)
            loss_domain = loss_gen
            self.is_adversarial = True
            tf.compat.v1.summary.scalar('loss/domain', loss_domain)
            tf.compat.v1.summary.scalar('loss/disc', self.loss_disc)

            tf.compat.v1.summary.histogram('loss/domain_logitsh_0',self.scoreh[...,0])
            tf.compat.v1.summary.histogram('loss/domain_logitsh_1',self.scoreh[...,1])
            tf.compat.v1.summary.histogram('loss/domain_logitsl_0', self.scorel[..., 0])
            tf.compat.v1.summary.histogram('loss/domain_logitsl_1', self.scorel[..., 1])

        elif self.args.domain == 'DANNc':
            assert 'late' in self.model, 'only late embedding implemented for now'
            zh, zl = self.Zh, self.Zl
            if self.args.lambda_reg > 0.0:
                label_reg = self.labelsh if self.hr_emb else self.labels
                if "vaihingen" in self.args.dataset:
                    label_reg = tf.expand_dims(label_reg[..., 0],-1)
                zh = tf.concat((zh,label_reg),axis=-1)
                zl = tf.concat((zl,label_reg),axis=-1)
            if self.args.lambda_reg < 1.0:
                label_sem = self.get_sem(self.labelsh) if self.hr_emb else self.get_sem(self.labels)
                label_sem = tf.cast(tf.expand_dims(label_sem,-1),tf.float32)
                zh = tf.concat((zh,label_sem),axis=-1)
                zl = tf.concat((zl,label_sem),axis=-1)

            self.scoreh = semi.domain_discriminator_small(zh)
            self.scorel = semi.domain_discriminator_small(zl)

            # Fake predictions towards 0, real preds towards 1
            loss_domain = cross_entropy(labels=tf.zeros_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scoreh) + \
                             cross_entropy(labels=tf.ones_like(self.scorel[...,0], dtype=tf.int32), logits=self.scorel)

            tf.compat.v1.summary.scalar('loss/domain', loss_domain)
            tf.compat.v1.summary.histogram('loss/domain_logitsh_0',self.scoreh[...,0])
            tf.compat.v1.summary.histogram('loss/domain_logitsh_1',self.scoreh[...,1])
            tf.compat.v1.summary.histogram('loss/domain_logitsl_0', self.scorel[..., 0])
            tf.compat.v1.summary.histogram('loss/domain_logitsl_1', self.scorel[..., 1])
        elif self.args.domain == 'L1':
            self.scoreh = semi.domain_discriminator(self.Zh)
            self.scorel = semi.domain_discriminator(self.Zl)

            loss_domain = tf.compat.v1.losses.absolute_difference(self.Zh,self.Zl)
            # loss_domain=0.0
            tf.compat.v1.summary.scalar('loss/domain', loss_domain)
        elif self.args.domain == 'Asso':
            self.ModelSemi = semi.SemisupModel()
            label_sem = self.get_sem(self.labelsh) if self.hr_emb else self.get_sem(self.labels)
            loss_domain = self.ModelSemi.add_semisup_loss(self.Zl,self.Zh,label_sem)
        elif self.args.domain == 'CDAN':
            if self.args.lambda_reg == 1.0:
                y_hat_ = self.y_hat['reg']
                y_hath_ = self.y_hath['reg']
            elif self.args.lambda_reg == 0.0:
                y_hat_ = self.y_hat['sem']
                y_hath_ = self.y_hath['sem']
            else:
                y_hat_ = tf.concat(self.y_hat['reg'],self.y_hat['sem'], axis=-1)
                y_hath_ = tf.concat(self.y_hath['reg'],self.y_hath['sem'], axis=-1)

            Zl = tools.batch_outerproduct(self.Zl, y_hat_, randomized=True)
            Zh = tools.batch_outerproduct(self.Zh, y_hath_, randomized=True)


            self.scoreh = semi.domain_discriminator_small(Zh)
            self.scorel = semi.domain_discriminator_small(Zl)

            loss_domain = cross_entropy(labels=tf.zeros_like(self.scoreh[..., 0], dtype=tf.int32), logits=self.scoreh) + \
                          cross_entropy(labels=tf.ones_like(self.scorel[..., 0], dtype=tf.int32), logits=self.scorel)

            tf.compat.v1.summary.scalar('loss/domain', loss_domain)
        elif self.args.domain == 'Contrast':
            if self.args.lambda_reg == 1.0:
                raise ValueError('not implemented for regresssion')

            # same class different domains
            loss_domain = tf.compat.v1.losses.mean_squared_error(self.Zl,self.Zh)

            # different class, different domain
            label_sem = self.get_sem(self.labelsh) if self.hr_emb else self.get_sem(self.labels)
            weights = [tf.expand_dims(tf.equal(label_sem, 0), -1),
                       tf.expand_dims(tf.equal(label_sem, 1), -1)]
            # zl[class == 1] and zh[class == 0]
            weights = [tf.broadcast_to(x, tf.shape(input=self.Zl)) for x in weights]
            #TODO check the effect 0.0 has
            zl1 = tf.compat.v1.where(weights[0], self.Zl,tf.zeros_like(self.Zl))
            zh0 = tf.compat.v1.where(weights[0], self.Zh,tf.zeros_like(self.Zh))

            d_sqrt = tools.pair_distance(zl1,zh0, randomized=False)
            loss_domain+= 0.5*tf.reduce_sum(input_tensor=tf.square(tf.maximum(0., 1.0 - d_sqrt)))

            # zl[class == 0] and zh[class == 1]
            zl0 = tf.compat.v1.where(weights[0], self.Zl, tf.zeros_like(self.Zl))
            zh1 = tf.compat.v1.where(weights[1], self.Zh, tf.zeros_like(self.Zl))

            d_sqrt = tools.pair_distance(zl0, zh1, randomized=False)
            loss_domain += 0.5 * tf.reduce_sum(input_tensor=tf.square(tf.maximum(0., 1.0 - d_sqrt)))
            tf.compat.v1.summary.scalar('loss/domain', loss_domain)

        else:
            raise ValueError('{} loss domain-transfer not defined'.format(self.args.domain))

        # self.loss += loss_domain*lambda_domain
        self.losses.append(loss_domain)
        self.scale_losses.append(lambda_domain)
    def add_distilled_loss(self):
        if self.args.lambda_reg == 1.0:
            y_hat_ = self.y_hat['reg']
            y_teacher = self.y_hat_teacher['reg']
        elif self.args.lambda_reg == 0.0:
            y_hat_ = self.y_hat['sem']
            y_teacher = self.y_hat_teacher['sem']
        else:
            y_hat_ = tf.concat((self.y_hat['reg'], self.y_hat['sem']), axis=-1)
            y_teacher = tf.concat((self.y_hat_teacher['reg'], self.y_hat_teacher['sem']), axis=-1)
        if self.args.is_hard_distill:
            assert self.args.lambda_reg == 0.0, 'implemented only for segmentation'
            y_teacher_hard = tf.math.argmax(y_hat_,axis=-1)
            loss_dst = cross_entropy(labels=y_teacher_hard,logits=y_hat_)
        else:
            loss_dst = tf.compat.v1.losses.mean_squared_error(y_hat_, y_teacher)
        lambda_dst = 1.0
        self.losses.append(loss_dst)
        self.scale_losses.append(lambda_dst)
        tf.compat.v1.summary.scalar('loss/distilled', loss_dst)

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
        inv_ = lambda x: inv_preprocess_tf(x, mean_train, scale_luminosity=scale, s2=not self.not_s2)
        max_ = 20. if self.args.dataset == 'palmage' else 2.0
        f1 = lambda x: tf.compat.v1.where(x == -1, x, x * (2.0 / max_dens))
        inv_regh_ = lambda x: uint8_(colorize(f1(x), vmin=-1, vmax=max_, cmap='viridis'))
        inv_reg_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=max_, cmap='viridis'))
        inv_sem_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=self.n_classes, cmap='jet'))
        inv_difreg_ = lambda x: uint8_(colorize(x,vmin=-2,vmax=2, cmap='coolwarm'))


        y_hat_reg = self.y_hat['reg']
        pred_class = tf.argmax(input=self.y_hat['sem'], axis=3)

        if self.hr_emb:
            y_hat_reg_down = tools.sum_pool(y_hat_reg, self.scale, name='y_reg_down')
            pred_class_down = tf.squeeze(tf.round(tools.avg_pool(self.float_(pred_class), self.scale, name='sem_down')), axis=3)
        else:
            y_hat_reg_down = y_hat_reg
            pred_class_down = pred_class

        feat_l_up = tf.map_fn(inv_, tools.bilinear(self.feat_l, size=self.patch_size * self.scale), dtype=tf.uint8)

        feat_l_ = tf.map_fn(inv_, self.feat_l, dtype=tf.uint8)

        feat_h_down = uint8_(
            tools.bilinear(self.feat_h, self.patch_size, name='HR_down')) if self.feat_h is not None else feat_l_

        if self.hr_emb:
            assert (self.labelsh.shape[1:3] == self.feat_h.shape[1:3])

            label_sem, w = self.get_sem(self.labelsh, return_w=True)
            image_array_top = self.concat_reg(self.labelsh, y_hat_reg, inv_regh_, inv_difreg_)
            image_array_mid = self.concat_sem(label_sem, pred_class, inv_sem_, inv_difreg_)
            image_array_bottom = tf.concat(axis=2, values=[feat_l_up, uint8_(self.feat_h), uint8_(tf.zeros_like(self.feat_h))])

            if self.args.lambda_reg == 1.0:
                image_array = tf.concat(axis=1, values=[image_array_top, image_array_bottom])
            elif self.args.lambda_reg == 0.0:
                image_array = tf.concat(axis=1, values=[image_array_mid, image_array_bottom])
            else:
                image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.compat.v1.summary.image('HR_Loss/HR', image_array, max_outputs=self.max_output_img)
            if self.is_hr_pred:
                metrics_reg = {
                    'metricsHR/mae': tf.compat.v1.metrics.mean_absolute_error(labels=self.get_reg(self.labelsh), predictions=y_hat_reg, weights=w),
                    'metricsHR/mse': tf.compat.v1.metrics.mean_squared_error(labels=self.get_reg(self.labelsh), predictions=y_hat_reg, weights=w),
                }

                metrics_sem = {
                    'metricsHR/iou': tf.compat.v1.metrics.mean_iou(labels=label_sem, predictions=pred_class, num_classes=2,
                                                       weights=w),
                    'metricsHR/prec': tf.compat.v1.metrics.precision(labels=label_sem, predictions=pred_class, weights=w),
                    'metricsHR/acc': tf.compat.v1.metrics.accuracy(labels=label_sem, predictions=pred_class, weights=w),
                    'metricsHR/recall': tf.compat.v1.metrics.recall(labels=label_sem, predictions=pred_class, weights=w)}

                if self.args.lambda_reg > 0.0:
                    self.eval_metric_ops.update(metrics_reg)
                if self.args.lambda_reg < 1.0:
                    self.eval_metric_ops.update(metrics_sem)

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

        if self.is_sr:
            image_array = tf.concat(axis=2, values=[feat_l_up, uint8_(self.HR_hat), uint8_(self.feat_h)])

            tf.compat.v1.summary.image('HR_hat-HR', image_array, max_outputs=self.max_output_img)

        args_tensor = tf.compat.v1.make_tensor_proto([([k, str(v)]) for k, v in sorted(self.args.__dict__.items())])
        meta = tf.compat.v1.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary = tf.compat.v1.Summary()
        summary.value.add(tag="FLAGS", metadata=meta, tensor=args_tensor)
        summary_writer = tf.compat.v1.summary.FileWriter(self.args.model_dir)
        summary_writer.add_summary(summary)

        if not self.is_training:
            # Compute evaluation metrics.
            labels = tf.expand_dims(self.labels[...,0],-1) if "vaihingen" in self.args.dataset else self.labels

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

            if self.is_sr:
                self.eval_metric_ops['metrics/semi_loss'] = tf.compat.v1.metrics.mean_squared_error(self.HR_hat, self.feat_h)
                self.eval_metric_ops['metrics/s2nr'] = tools.snr_metric(self.HR_hat, self.feat_h)
        else:
            self.eval_metric_ops = None

    def get_reg(self,labels):
        if "vaihingen" in self.args.dataset:
            labels = tf.expand_dims(labels[...,0],axis=-1)
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
        if self.not_s2:
            # assert not self.args.dataset == 'vaihingen', 'implement for missing points with -1'
            x = tf.clip_by_value(labels[...,1],0,10000)
            x = tools.avg_pool(x, scale)
            xm = tools.max_pool(labels[...,1,tf.newaxis], scale)

            lb_reg = tf.compat.v1.where(tf.equal(xm,-1),xm,x, name='lb_reg_down')
            lb_sem = tools.median_pool(labels[...,0], scale, name='lb_sem_down')
            return tf.concat((lb_sem,lb_reg),axis=3)

        else:
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
        elif self.args.optimizer == 'adabound':
            optimizer = AdaBoundOptimizer(learning_rate=self.args.lr, final_lr=10*self.args.lr)
        elif self.args.optimizer == 'adaboundA':
            optimizer = AdaBoundOptimizer(learning_rate=self.args.lr, final_lr=self.args.lr)
        elif self.args.optimizer == 'adaboundB':
            optimizer = AdaBoundOptimizer(learning_rate=self.args.lr, final_lr=0.1*self.args.lr)
        elif self.args.optimizer == 'SGD':
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == 'SGDa':
            progress = tools.get_progress(self.args)
            lr = tf.compat.v1.where(tf.less(progress,0.8), 1.0, 0.1)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.args.lr*lr)
        elif self.args.optimizer == 'SGDb':
            progress = tools.get_progress(self.args)
            lr = tf.compat.v1.where(tf.less(progress,0.7), 1.0, 0.1)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.args.lr*lr)
        elif self.args.optimizer == 'SGDc':
            progress = tools.get_progress(self.args)
            lr = tf.compat.v1.where(tf.less(progress,0.6), 1.0, 0.1)
            optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.args.lr*lr)
        elif self.args.optimizer == 'momentum':
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.args.lr, momentum=0.9, use_nesterov=True)
        elif self.args.optimizer == 'annealing':
            learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.001, power=0.75)
            tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif self.args.optimizer == 'annealing10':
            learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.01, power=0.75)
            tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif self.args.optimizer == 'annealing100':
            learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.1, power=0.75)
            tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif self.args.optimizer == 'annealing0':
            learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.0001, power=0.75)
            tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif self.args.optimizer == 'annealing00':
            learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.00001, power=0.75)
            tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
            optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
        elif self.args.optimizer == 'annealingA':
            progress = tools.get_progress(self.args)
            learning_rate = tools.inv_lr_decay(self.args.lr, progress, gamma=10.0, power=0.75)
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
                # if self.args.semi is not None and not 'Rev' in self.args.semi:
                if self.is_adversarial:
                    train_d = optimizer.minimize(self.loss_disc, global_step=step, var_list=self.vars_d)
                    train_g = optimizer.minimize(self.loss, global_step=step, var_list=self.vars_g)
                    self.train_op = tf.cond(pred=tf.equal(0, tf.cast(tf.math.mod(step, self.args.gen_loss_every), dtype=tf.int32)),
                                            true_fn=lambda: train_g,
                                            false_fn=lambda: train_d)
                else:
                    self.train_op = optimizer.minimize(self.loss, global_step=step, var_list=vars_train)
                if self.args.optimizer == 'adam':
                    lr_adam = tools.get_lr_ADAM(optimizer, learning_rate=self.args.lr)
                    tf.compat.v1.summary.scalar('loss/adam_lr', tf.math.log(lr_adam))
            else:
                train_op1 = optimizer.minimize(self.lossTasks, global_step=step,var_list=vars_train)
                train_op2 = optimizer.minimize(self.loss_w, global_step=step, var_list=vars_train)
                self.train_op = tf.cond(pred=tf.equal(0, tf.cast(tf.math.mod(step, self.args.l2_weights_every), dtype=tf.int32)),
                                        true_fn=lambda: train_op1,
                                        false_fn=lambda: train_op2)
            # if 'Asso' in self.args.domain:
            #     self.ModelSemi.add_vars()
