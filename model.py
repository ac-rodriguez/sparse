import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import sys
import numpy as np
cross_entropy = tf.losses.sparse_softmax_cross_entropy
import tensorflow.contrib.slim as slim

from colorize import colorize, inv_preprocess_tf
from models_reg import simple, countception
import models_sr as sr
from tools_tf import bilinear, snr_metric, sum_pool, avg_pool, max_pool, get_lr_ADAM
import models_semi as semi
class Model:
    def __init__(self, params):
        self.args = params['args']
        self.model_dir = params['model_dir']

        self.scale = self.args.scale
        self.is_hr_label = self.args.is_hr_label
        if self.args.is_fake_hr_label:
            self.is_hr_label = True
        if self.args.is_bilinear:
            self.down_ = lambda x, _: bilinear(x, self.patch_size, name='HR_hat_down')
            self.up_ = lambda x, _: bilinear(x, self.patch_size * self.scale, name='HR_hat_down')
        else:
            self.down_ = lambda x, ch: tf.layers.conv2d(x, ch, 3, strides=self.scale, padding='same')
            self.up_ = lambda x, ch: tf.layers.conv2d_transpose(x, ch, 3, strides=self.scale, padding='same')

        self.float_ = lambda x: tf.cast(x, dtype=tf.float32)

        self.sem_threshold = 0
        self.max_output_img = 1
        self.model = self.args.model
        self.two_ds = True

        self.loss_in_HR = False
        self.add_yhath = False
        self.add_yhat = True

        self.is_domain_transfer = False
        self.is_semi = False
        self.is_sr = False
        self.sr_on_labeled = True

        self.pad = self.args.sq_kernel*16//2
        if ('HR' in self.model or 'SR' in self.model or 'DA_h' in self.model) and not '_l' in self.model and self.is_hr_label:
            self.loss_in_HR = True

    def model_fn(self, features, labels, mode):
        self.is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # self.features = features
        self.labels = labels

        if isinstance(features, dict):
            self.feat_l, self.feat_h = features['feat_l'], features['feat_h']
        else:
            self.feat_l = features
            self.feat_h = None
        if self.two_ds:
            self.feat_lU, self.feat_hU = features['feat_lU'], features['feat_hU']
        self.patch_size = self.feat_l.shape[1]
        if self.args.is_fake_hr_label:
            self.labels = tf.image.resize_nearest_neighbor(self.labels, size=(self.patch_size*self.scale, self.patch_size*self.scale)) / (self.scale**2)
        if self.args.is_same_volume:
            self.config = {'hr':self.is_hr_label,'scale':self.scale,'patch':self.patch_size}
        else:
            self.config = None

        self.compute_predicitons()

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=self.y_hat)
        # if self.loss_in_HR:
        if self.args.sq_kernel is None:
            self.sem_threshold = 1e-5
        self.compute_loss()
        if self.loss_in_HR and not self.is_sr:
            self.compute_summaries(self.y_hath)
        else:
            self.compute_summaries(self.y_hat)


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

    def compute_predicitons(self):
        size = self.patch_size * self.args.scale
        # Baseline Models
        if self.model == 'simple':  # Baseline  No High Res for training
            self.y_hat = simple(self.feat_l, n_channels=1, is_training=self.is_training)
        elif self.model == 'count' or self.model == 'counth':
            self.y_hat, self.Zl = countception(self.feat_l,pad=self.pad, is_training=self.is_training, is_return_feat=True,  config_volume=self.config)
            if self.two_ds:
                    _, self.ZlU = countception(self.feat_lU,pad=self.pad, is_training=self.is_training, is_return_feat=True,  config_volume=self.config)

            if self.model == 'counth':  # using down-sampled HR images too
                self.add_yhath = True
                feat_h_down = self.down_(self.feat_h,3)
                feat_h_down = tf.layers.conv2d(feat_h_down,self.feat_l.shape[-1],kernel_size=1,strides=1)
                self.y_hath, self.Zh = countception(feat_h_down, pad=self.pad, is_training=self.is_training,
                                                   is_return_feat=True)

        elif self.model == 'countSR' or self.model == 'countSRu':
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
                    self.y_hat[key] = bilinear(val, self.patch_size)
            else:
                assert self.loss_in_HR == True
        elif self.model == 'countSRs':
            self.is_sr = True
            self.HR_hat = semi.decode(self.feat_l, scale=self.scale, is_bn=True, is_training=self.is_training, n_feat_last=3)
            self.HR_hatU = semi.decode(self.feat_lU, scale=self.scale, is_bn=True, is_training=self.is_training, n_feat_last=3)

            feat_l_up = self.up_(self.feat_l, 8)
            feat = tf.concat([self.HR_hat, feat_l_up], axis=3)

            self.y_hat = countception(feat,pad=self.pad, is_training=self.is_training, config_volume=self.config)
            if not self.is_hr_label: # added as a baseline
                for key, val in self.y_hat.iteritems():
                    self.y_hat[key] = bilinear(val, self.patch_size)
            else:
                assert self.loss_in_HR == True
        elif self.model == 'countSR_l':
            self.is_sr = True
            self.HR_hat = sr.SR_task(feat_l=self.feat_l, size=size, is_batch_norm=True, is_training=self.is_training)
            self.HR_hatU = sr.SR_task(feat_l=self.feat_lU, size=size, is_batch_norm=True, is_training=self.is_training)

            feat = semi.encode(self.HR_hat, is_training=self.is_training, is_bn=True)
            feat = tf.concat([feat, self.feat_l], axis=3)
            self.y_hat = countception(feat,pad=self.pad, is_training=self.is_training)
        elif self.model == 'countHR':
            feat_l_up = self.up_(self.feat_l, 8)
            feat = tf.concat([self.feat_h, feat_l_up[..., 3:]], axis=3)
            self.y_hat , self.Zh = countception(feat,pad=self.pad, is_training=self.is_training, is_return_feat=True)
            if self.two_ds:
                feat_lU_up = self.up_(self.feat_lU, 8)
                feat = tf.concat([self.feat_hU, feat_lU_up[..., 3:]], axis=3)
                _, self.ZhU = countception(feat,pad=self.pad, is_training=self.is_training, is_return_feat=True)

            assert self.loss_in_HR == True
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
        elif self.model == 'countHR_lb': # Using HR channels only

            Ench = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale = self.scale)
            self.y_hat, self.Zh = countception(Ench,pad=self.pad, is_training=self.is_training, is_return_feat=True)
            if self.two_ds:
                EnchU = semi.encode(input=self.feat_hU, is_training=self.is_training, is_bn=True, scale=self.scale)
                _, self.ZhU = countception(EnchU, pad=self.pad, is_training=self.is_training, is_return_feat=True)
        elif 'DA_h' in self.model:
            self.daH_models()

        elif 'DA' in self.model:
            self.daL_models()

        else:
            print('Model {} not defined'.format(self.model))
            sys.exit(1)
        if self.args.is_out_relu:
            self.y_hat['reg'] = tf.nn.relu(self.y_hat['reg'])
        if self.is_domain_transfer:
            Emb = tf.concat((self.Zh, self.Zl), axis=0, name='Embeddings')
            Label = tf.concat((tf.ones(tf.shape(self.Zh)[:-1]), tf.zeros(tf.shape(self.Zh)[:-1])), axis=0,
                              name='EmbeddingsLabel')

    def compute_loss(self):

        int_ = lambda x: tf.cast(x, dtype=tf.int32)
        if self.is_hr_label and not self.loss_in_HR:
            self.labels = self.compute_labels_ls(self.labels, self.scale)

        self.w = self.float_(tf.where(tf.greater_equal(self.labels, 0.0),
                                      tf.ones_like(self.labels),  ## for wherever i have labels
                                      tf.zeros_like(self.labels)))
        self.label_sem = tf.squeeze(int_(tf.greater(self.labels, self.sem_threshold)), axis=3)

        self.lossTasks = 0.0
        if self.add_yhat:
            loss_reg = tf.losses.mean_squared_error(labels=self.labels, predictions=self.y_hat['reg'], weights=self.w)
            loss_sem = cross_entropy(labels=self.label_sem, logits=self.y_hat['sem'], weights=self.w)

            loss_ = self.args.lambda_reg * loss_reg + (1.0 - self.args.lambda_reg) * loss_sem
            tf.summary.scalar('loss/reg', loss_reg)
            tf.summary.scalar('loss/sem', loss_sem)

            self.lossTasks+=loss_
        if self.add_yhath:
            loss_reg = tf.losses.mean_squared_error(labels=self.labels, predictions=self.y_hath['reg'], weights=self.w)
            loss_sem = cross_entropy(labels=self.label_sem, logits=self.y_hath['sem'], weights=self.w)

            loss_ = self.args.lambda_reg * loss_reg + (1.0 - self.args.lambda_reg) * loss_sem

            self.lossTasks +=loss_
        # SR loss
        if self.is_sr and self.args.lambda_sr > 0:
            if self.args.sr_after is not None:
                if self.args.sr_after > 0:
                    w1 = tf.where(tf.greater(self.args.sr_after, tf.train.get_global_step()), 0., 1.)
                else:
                    w1 = tf.where(tf.greater(self.args.sr_after, tf.train.get_global_step()), 1., 0.)
            else:
                w1 = 1.
            if self.sr_on_labeled:
                loss_sr = tf.nn.l2_loss(self.HR_hat - self.feat_h)
            else:
                loss_sr = tf.nn.l2_loss(self.HR_hatU - self.feat_hU)
            tf.summary.scalar('loss/SR', loss_sr)
            self.lossTasks += self.args.lambda_sr * w1 * loss_sr

        # L2 weight Regularizer
        W = [v for v in tf.trainable_variables() if ('weights' in v.name or 'kernel' in v.name)]
        # Lambda_weights is always rescaled with 0.0005
        l2_weights = tf.add_n([0.0005 * tf.nn.l2_loss(v) for v in W])
        tf.summary.scalar('loss/L2Weigths', l2_weights)
        self.loss_w = self.args.lambda_weights * l2_weights

        self.loss = self.lossTasks + self.loss_w
        if self.is_semi:
            self.add_semi_loss(self.y_hat)
        if self.args.domain is not None:
            self.add_domain_loss()
        grads = tf.gradients(self.loss, W, name='gradients')
        norm = tf.add_n([tf.norm(g, name='norm') for g in grads])

        tf.summary.scalar('loss/L2Grad', norm)

    def daH_models(self):
        assert self.loss_in_HR
        self.is_domain_transfer = True
        self.add_yhath = True
        if self.model == 'countDA_hmid': #self.model == 'countHR_ld':
            # Train on (HR,y) and (Lr,y). Test on (Lr,y). Comparison on Higher-level features


            encHR = semi.encode_same(input=self.feat_h, is_training=self.is_training, is_bn=True)
            encLR = semi.decode(self.feat_l, is_training=self.is_training, is_bn=True, scale=self.scale)

            self.y_hat, self.Zl = countception(encLR, pad=self.pad, is_training=self.is_training, is_return_feat=True,  config_volume=self.config)
            self.y_hath, self.Zh = countception(encHR, pad=self.pad, is_training=self.is_training, is_return_feat=True, config_volume=self.config)

        elif self.model == 'countDA_hlate': #self.model == 'countHR_le':
            # Train on (HR,y) and (Lr,y). Test on (Lr,y). Comparison on (last) Higher-level features

            encHR = semi.encode_same(input=self.feat_h, is_training=self.is_training, is_bn=True)
            encLR = semi.decode(self.feat_l, is_training=self.is_training, is_bn=True, scale=self.scale)

            self.y_hat, self.Zl = countception(encLR, pad=self.pad, is_training=self.is_training, is_return_feat=True, last=True,  config_volume=self.config)
            self.y_hath, self.Zh = countception(encHR, pad=self.pad, is_training=self.is_training, is_return_feat=True, last=True,  config_volume=self.config)
        else:
            print('Model {} not defined'.format(self.model))
            sys.exit(1)

    def daL_models(self):
        self.is_domain_transfer = True
        self.add_yhath = True
        if self.model == 'countDAearly':  # self.model == 'countHR_lc':
            # Train on (HR,y) and (Lr,y). Test on (Lr,y). Comparison on Lower-level features

            self.Zh = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            self.Zl = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)

            self.y_hat = countception(self.Zl, pad=self.pad, is_training=self.is_training)
            self.y_hath = countception(self.Zh, pad=self.pad, is_training=self.is_training)

        elif self.model == 'countDAmid': #self.model == 'countHR_ld':
            # Train on (HR,y) and (Lr,y). Test on (Lr,y). Comparison on Higher-level features

            encHR = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            encLR = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)

            self.y_hat, self.Zl = countception(encLR, pad=self.pad, is_training=self.is_training, is_return_feat=True,  config_volume=self.config)
            self.y_hath, self.Zh = countception(encHR, pad=self.pad, is_training=self.is_training, is_return_feat=True,  config_volume=self.config)
        elif self.model == 'countDAlate': #self.model == 'countHR_le':
            # Train on (HR,y) and (Lr,y). Test on (Lr,y). Comparison on (last) Higher-level features

            encHR = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            encLR = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)

            self.y_hat, self.Zl = countception(encLR, pad=self.pad, is_training=self.is_training, is_return_feat=True, last=True,  config_volume=self.config)
            self.y_hath, self.Zh = countception(encHR, pad=self.pad, is_training=self.is_training, is_return_feat=True, last=True,  config_volume=self.config)
        elif self.model == 'countDApair':
            # Train on (HR,y). Test on (Lr,y). LR is paired with HR. Comparison on Higher-level features

            encHR = semi.encode(input=self.feat_h, is_training=self.is_training, is_bn=True, scale=self.scale)
            encLR = semi.encode_same(self.feat_l, is_training=self.is_training, is_bn=True)

            self.y_hat, self.Zl = countception(encLR, pad=self.pad, is_training=self.is_training,
                                          is_return_feat=True)
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
            print('Model {} not defined'.format(self.model))
            sys.exit(1)


    def add_semi_loss(self, y_hat):

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

            tf.summary.scalar('loss/domain_semi', loss_domain)

            self.loss += loss_domain
        else:
            if self.args.semi == 'semi':
                self.fake_ = semi.discriminator(y_hat['reg'])
                self.real_ = semi.discriminator(self.labels)
                # Fake predictions towards 0, real preds towards 1
                self.loss_disc = cross_entropy(labels=tf.zeros_like(self.labels, dtype=tf.int32),logits=self.fake_) + \
                            cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32), logits=self.real_,weights=self.w)

                self.loss_gen = cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32),logits=self.fake_)
            elif self.args.semi == 'semiFeat':
                self.fake_,feat_f = semi.discriminator(y_hat['reg'],return_feat=True)
                self.real_,feat_r = semi.discriminator(self.labels, return_feat=True)
                # Fake predictions towards 0, real preds towards 1
                self.loss_disc = cross_entropy(labels=tf.zeros_like(self.labels, dtype=tf.int32),logits=self.fake_) + \
                            cross_entropy(labels=tf.ones_like(self.labels,dtype=tf.int32), logits=self.real_,weights=self.w)
                self.loss_gen = tf.losses.mean_squared_error(labels=feat_r, predictions=feat_f)

            elif self.args.semi == 'semi1':
                self.fake_ = semi.discriminator(tf.concat((y_hat['reg'],y_hat['sem']),axis=-1))
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
                self.loss_gen = tf.losses.mean_squared_error(labels=feat_r, predictions=feat_f)

            else:
                print('{} model semi-supervised not defined'.format(self.args.semi))
                sys.exit(1)
            tf.summary.scalar('loss/gen', self.loss_gen)
            tf.summary.scalar('loss/disc', self.loss_disc)

            self.loss+= self.loss_gen

    def add_domain_loss(self):
        assert self.is_domain_transfer, ' domain-transfer not defined for model:{} '.format(self.model)

        if self.args.domain == 'domainRev' or self.args.domain == 'domainRevL':

            self.scoreh = semi.domain_discriminator(self.Zh)
            self.scorel = semi.domain_discriminator(self.Zl)

            # Fake predictions towards 0, real preds towards 1
            loss_domain = cross_entropy(labels=tf.zeros_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scoreh) + \
                             cross_entropy(labels=tf.ones_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scorel)
            if 'L' in self.args.domain:
                loss_domain+= tf.losses.absolute_difference(self.Zh, self.Zl)

            tf.summary.scalar('loss/domain', loss_domain)
        elif self.args.domain == 'domainRevs':

            self.scoreh = semi.domain_discriminator_small(self.Zh)
            self.scorel = semi.domain_discriminator_small(self.Zl)

            # Fake predictions towards 0, real preds towards 1
            loss_domain = cross_entropy(labels=tf.zeros_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scoreh) + \
                             cross_entropy(labels=tf.ones_like(self.scoreh[...,0], dtype=tf.int32), logits=self.scorel)

            tf.summary.scalar('loss/domainS', loss_domain)
        elif self.args.domain == 'domainL1':
            self.scoreh = semi.domain_discriminator(self.Zh)
            self.scorel = semi.domain_discriminator(self.Zl)

            loss_domain = tf.losses.absolute_difference(self.Zh,self.Zl)
            # loss_domain=0.0
            tf.summary.scalar('loss/domain', loss_domain)
        elif self.args.domain == 'domainAsso':
            self.ModelSemi = semi.SemisupModel()
            loss_domain = self.ModelSemi.add_semisup_loss(self.Zl,self.Zh,self.label_sem)
        else:
            print('{} loss domain-transfer not defined'.format(self.args.domain))
            sys.exit(1)

        self.loss += loss_domain

    def compute_summaries(self, y_hat):
        graph = tf.get_default_graph()
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
        inv_ = lambda x: inv_preprocess_tf(x, mean_train, scale_luminosity=scale)
        if self.loss_in_HR:
            f1 = lambda x: tf.where(x == -1, x, x * (2.0 / max_dens))
            inv_reg_ = lambda x: uint8_(colorize(f1(x), vmin=-1, vmax=2.0, cmap='jet'))
        else:
            inv_reg_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=2.0, cmap='jet'))
        inv_sem_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=1.0, cmap='hot'))
        int_ = lambda x: tf.cast(x, dtype=tf.int32)
        inv_difreg_ = lambda x: uint8_(colorize(x,vmin=-2,vmax=2, cmap='coolwarm'))

        pred_class = tf.argmax(y_hat['sem'], axis=3)
        w = self.w
        labels = self.labels
        label_sem = tf.where(tf.squeeze(tf.greater(self.w, 0), axis=3), self.label_sem,
                             tf.ones_like(self.label_sem, dtype=tf.int32) * -1)
        y_hat_reg = y_hat['reg']

        feat_l_up = tf.map_fn(inv_, bilinear(self.feat_l, size=self.patch_size * self.scale), dtype=tf.uint8)

        feat_l_ = tf.map_fn(inv_, self.feat_l, dtype=tf.uint8)

        feat_h_down = uint8_(
            bilinear(self.feat_h, self.patch_size, name='HR_down')) if self.feat_h is not None else feat_l_

        image_array_top = tf.map_fn(inv_reg_, tf.concat(axis=2, values=[labels, y_hat_reg, ]), dtype=tf.uint8)
        image_array_mid = tf.map_fn(inv_sem_, tf.concat(axis=2, values=[label_sem, int_(pred_class)]), dtype=tf.uint8)

        image_array_top = tf.concat(axis=2, values=[image_array_top,
                                                    tf.map_fn(inv_difreg_, labels - y_hat_reg, dtype=tf.uint8)])
        image_array_mid = tf.concat(axis=2, values=[image_array_mid,
                                                    tf.map_fn(inv_difreg_, label_sem - int_(pred_class),
                                                              dtype=tf.uint8)])

        if self.loss_in_HR:
            assert (self.labels.shape[1:3] == self.feat_h.shape[1:3])

            image_array_bottom = tf.concat(axis=2, values=[feat_l_up, uint8_(self.feat_h),uint8_(self.feat_h)])
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.summary.image('HR_Loss/HR', image_array, max_outputs=self.max_output_img)

            # Compute summaries in LR space

            labels = self.compute_labels_ls(self.labels, self.scale)
            y_hat_reg = sum_pool(y_hat['reg'], self.scale, name='y_reg_down')

            w = self.float_(tf.where(tf.greater_equal(labels, 0.0),
                                     tf.ones_like(labels),  ## for wherever i have labels
                                     tf.zeros_like(labels)))

            label_sem = tf.squeeze(int_(tf.greater(labels, self.sem_threshold)), axis=3)
            label_sem = tf.where(tf.squeeze(tf.equal(w, 1), axis=3), label_sem,
                                 tf.ones_like(label_sem, dtype=tf.int32) * -1)

            pred_class = tf.squeeze(tf.round(avg_pool(self.float_(pred_class), self.scale, name='sem_down')), axis=3)

            inv_reg_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=2.0, cmap='jet'))
            # inv_difreg_ = lambda x: uint8_(colorize(x, vmin=-1, vmax=2.0, cmap='hot'))

            image_array_top = tf.map_fn(inv_reg_, tf.concat(axis=2, values=[labels, y_hat_reg ]), dtype=tf.uint8)
            image_array_mid = tf.map_fn(inv_sem_, tf.concat(axis=2, values=[label_sem, int_(pred_class)]),dtype=tf.uint8)

            image_array_top = tf.concat(axis=2,values=[image_array_top,tf.map_fn(inv_difreg_, labels-y_hat_reg , dtype=tf.uint8)])
            image_array_mid = tf.concat(axis=2, values=[image_array_mid, tf.map_fn(inv_difreg_,  label_sem-int_(pred_class), dtype=tf.uint8) ])

            image_array_bottom = tf.concat(axis=2, values=[feat_l_, feat_h_down, feat_h_down])

            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.summary.image('HR_Loss/LR', image_array, max_outputs=self.max_output_img)

        else:

            image_array_bottom = tf.concat(axis=2, values=[feat_l_, feat_h_down, feat_h_down])
            image_array = tf.concat(axis=1, values=[image_array_top, image_array_mid, image_array_bottom])

            tf.summary.image('LR_Loss/LR', image_array, max_outputs=self.max_output_img)

        if self.is_sr:
            image_array = tf.concat(axis=2, values=[feat_l_up, uint8_(self.HR_hat), uint8_(self.feat_h)])

            tf.summary.image('HR_hat-HR', image_array, max_outputs=self.max_output_img)

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
                'metrics/mae': tf.metrics.mean_absolute_error(labels=labels, predictions=y_hat_reg, weights=w),
                'metrics/mse': tf.metrics.mean_squared_error(labels=labels, predictions=y_hat_reg, weights=w),
                'metrics/iou': tf.metrics.mean_iou(labels=label_sem, predictions=pred_class, num_classes=2, weights=w),
                'metrics/prec': tf.metrics.precision(labels=label_sem, predictions=pred_class, weights=w),
                'metrics/acc': tf.metrics.accuracy(labels=label_sem, predictions=pred_class, weights=w),
                'metrics/recall': tf.metrics.recall(labels=label_sem, predictions=pred_class, weights=w)}

            if self.is_sr:
                self.eval_metric_ops['metrics/semi_loss'] = tf.metrics.mean_squared_error(self.HR_hat, self.feat_h)
                self.eval_metric_ops['metrics/s2nr'] = snr_metric(self.HR_hat, self.feat_h)
        else:
            self.eval_metric_ops = None

    @staticmethod
    def compute_labels_ls(labels,scale):
        x = tf.clip_by_value(labels,0,1000)
        x = sum_pool(x, scale)

        xm = max_pool(labels, scale)
        x = tf.where(tf.equal(xm,-1),xm,x, name='Label_down')
        return x


    def compute_train_op(self):

        if self.args.optimizer == 'adagrad':
            optimizer = tf.train.AdagradOptimizer(learning_rate=self.args.lr)
        elif self.args.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr)
        else:
            print('option not defined')
            sys.exit(1)
        slim.model_analyzer.analyze_vars(
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            step = tf.train.get_global_step()
            if self.args.l2_weights_every is None:
                if self.args.semi is not None and not 'Rev' in self.args.semi:
                    train_g = optimizer.minimize(self.loss, global_step=step)
                    train_d = optimizer.minimize(self.loss_gen, global_step=step)
                    self.train_op = tf.cond(tf.equal(0, tf.to_int32(tf.mod(step, 2))),
                                            true_fn=lambda: train_g,
                                            false_fn=lambda: train_d)
                else:
                    self.train_op = optimizer.minimize(self.loss, global_step=step)
                if self.args.optimizer == 'adam':
                    lr_adam = get_lr_ADAM(optimizer, learning_rate=0.01)
                    tf.summary.scalar('loss/adam_lr', lr_adam)
            else:
                train_op1 = optimizer.minimize(self.lossTasks, global_step=step)
                train_op2 = optimizer.minimize(self.loss_w, global_step=step)
                self.train_op = tf.cond(tf.equal(0, tf.to_int32(tf.mod(step, self.args.l2_weights_every))),
                                        true_fn=lambda: train_op1,
                                        false_fn=lambda: train_op2)
            # if 'Asso' in self.args.domain:
            #     self.ModelSemi.add_vars()
