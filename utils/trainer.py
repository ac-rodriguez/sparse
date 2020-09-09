import tensorflow as tf
import numpy as np
cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

# from AdaBound import AdaBoundOptimizer

from utils.colorize import colorize, inv_preprocess_tf
# from utils.models_reg import simple, countception

from utils.models_reg import SimpleA
import utils.tools_tf as tools
# import utils.models_semi as semi
from utils.location_encoder import SpatialModel

def tf_function():
    def decorator(func):
        if tf.__version__.startswith('2.'):
            return tf.function(func)
        else:
            return func
    return decorator

class Trainer():

    def __init__(self, args, inference_only = False):
        super(Trainer, self).__init__()
        self.args = args
        self.model_dir = args.model_dir

        self.scale = 1
        self.is_slim = self.args.is_slim_eval
        self.is_dropout = self.args.is_dropout_uncertainty

        self.float_ = lambda x: tf.cast(x, dtype=tf.float32)

        self.sem_threshold = 0
        self.max_output_img = 1
        self.model_name = self.args.model
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

        if 'simpleA' in self.model_name:
            depth = self.model_name.replace('simpleA', '')
            depth = 0 if depth == '' else int(depth)
            self.model = SimpleA(self.n_classes,extra_depth=depth,lambda_reg=args.lambda_reg, is_dropout=self.is_dropout)
        else:
            raise NotImplementedError

        if self.args.is_use_location:
            self.model = SpatialModel(args,image_model=self.model, fusion=self.args.fusion_type)

        if not inference_only:
            self.mse_loss = tf.keras.losses.MeanSquaredError()
            self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


            self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
            self.train_ce = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
            self.train_mse = tf.keras.metrics.MeanSquaredError()
            self.train_mae = tf.keras.metrics.MeanAbsoluteError()

            self.val_metrics_reg, self.val_metrics_sem = self.define_val_metrics()

            # adding metrics per gt file
            gt_files = [x['gt'].split('/')[-2:] for x in self.args.val]
            gt_files = {'/'.join(x) for x in gt_files}
            for gt_ in gt_files:
                reg_, sem_ = self.define_val_metrics(prefix=gt_+'_')
                self.val_metrics_reg = {**self.val_metrics_reg,**reg_}
                self.val_metrics_sem = {**self.val_metrics_sem,**sem_}

            # add prec and recall
            self.define_optimiter()

            # Define summary locs:

            self.train_writer = tf.summary.create_file_writer(self.model_dir)
            
            self.val_dirname = self.model_dir+'/val'
            if not self.args.is_train:
                self.val_dirname = self.val_dirname+ "only"
            if self.args.is_dropout_uncertainty:
                self.val_dirname = f'{self.val_dirname}drop{self.args.n_eval_dropout}'            
            self.val_writer = tf.summary.create_file_writer(self.val_dirname)

    def define_val_metrics(self, prefix='val_'):
        metrics_dict_reg = {}
        metrics_dict_reg[prefix+'mse'] = tf.keras.metrics.MeanSquaredError()
        metrics_dict_reg[prefix+'mae'] = tf.keras.metrics.MeanAbsoluteError()

        metrics_dict_sem = {}
        metrics_dict_sem[prefix+'ce'] = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
        metrics_dict_sem[prefix+'acc'] = tf.keras.metrics.SparseCategoricalAccuracy()
        metrics_dict_sem[prefix+'iou'] = tf.keras.metrics.MeanIoU(self.n_classes+1)
        metrics_dict_sem[prefix+'prec'] = tf.keras.metrics.Precision()
        metrics_dict_sem[prefix+'rec'] = tf.keras.metrics.Recall()

        return metrics_dict_reg, metrics_dict_sem

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

    @tf_function()
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(features, is_training=True)
            loss = self.compute_loss(predictions, labels)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions

    # @tf.function
    def test_step(self, features, labels):
        predictions = self.model(features, is_training=False)
        self.update_sum_val(predictions,labels)

    @tf_function()
    def compute_loss(self, predictions, labels):
        losses = []

        label_sem, w = self.get_sem(labels, return_w=True)

        lam_evol = tools.evolving_lambda(self.args,
                                         height=self.args.low_task_evol) if self.args.low_task_evol is not None else 1.0
        if self.args.lambda_reg > 0.0:
            w_reg = self.get_w(labels, is_99=True)
            loss_reg = self.mse_loss(y_true=labels, y_pred=predictions['reg'],
                                                              sample_weight=w_reg)

            losses.append((loss_reg, self.args.lambda_reg * lam_evol))

            # self.lossTasks+= self.args.lambda_reg * loss_reg * lam_evol
            # tf.summary.scalar('loss/reg', loss_reg)

        if self.args.lambda_reg < 1.0:
            w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
            loss_sem = self.ce_loss(y_true=label_sem, y_pred=predictions['sem'], sample_weight=w_)
            # self.losses.append(loss_sem)
            losses.append((loss_sem, (1.0 - self.args.lambda_reg) * lam_evol))
            # self.lossTasks+= (1.0 - self.args.lambda_reg) * loss_sem * lam_evol
            # tf.compat.v1.summary.scalar('loss/sem', loss_sem)

        if self.args.wdecay >0.0:

            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in self.model.variables
                    if 'bias' not in v.name ])
            losses.append((lossL2,self.args.wdecay))

        loss_with_scales = [a * b for a, b in losses]


        loss = tf.reduce_sum(loss_with_scales)

        return loss

    def update_sum_train(self,labels, predictions):
        label_sem, w = self.get_sem(labels, return_w=True)

        if self.args.lambda_reg > 0.0:
            w_reg = self.get_w(labels, is_99=True)
            self.train_mse(y_true=labels, y_pred=predictions['reg'],
                                                              sample_weight=w_reg)
            self.train_mae(y_true=labels, y_pred=predictions['reg'],
                                                              sample_weight=w_reg)

        if self.args.lambda_reg < 1.0:
            w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
            self.train_ce(y_true=label_sem, y_pred=predictions['sem'], sample_weight=w_)
    
        # @tf.function
    def summaries_train(self, step):
        with self.train_writer.as_default():
            if self.args.lambda_reg > 0.0:
                assert not (np.isnan(self.train_mse.result().numpy()) or np.isnan(self.train_mae.result().numpy()))
                tf.summary.scalar('train/mse',self.train_mse.result(),step=step)
                tf.summary.scalar('train/mae',self.train_mae.result(),step=step)
            if self.args.lambda_reg < 1.0:
                assert not np.isnan(self.train_ce.result().numpy())
                tf.summary.scalar('train/ce',self.train_ce.result(),step=step)
            tf.summary.scalar('train/lr',self.optimizer._decayed_lr(tf.float32),step=step)
    
    def reset_sum_train(self):
        self.train_mse.reset_states()
        self.train_mae.reset_states()
        self.train_ce.reset_states()

    def update_sum_val(self, predictions, labels, info=None):
        
        label_sem, w = self.get_sem(labels, return_w=True)
        if info is not None:
            info = list(info.numpy())
            info = [x.decode("utf-8") for x in info]

        if self.args.lambda_reg > 0.0:
            w_reg = self.get_w(labels, is_99=True)
            for key in self.val_metrics_reg.keys():
                if key.startswith('val_'):
                    self.val_metrics_reg[key](y_true=labels, y_pred=predictions['reg'],
                                                                sample_weight=w_reg)
                elif info is not None:
                    key_ = key.replace('/','_').rsplit('_',1)[0]
                    if key_ in info:
                        info_bool = [key_ == x for x in info]
                        info_bool = np.array(info_bool).reshape(-1,1,1,1)

                        self.val_metrics_reg[key](y_true=labels, y_pred=predictions['reg'],
                                                sample_weight=w_reg*info_bool)

        if self.args.lambda_reg < 1.0:
            w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
            pred_class = tf.argmax(input=predictions['sem'], axis=3)
            for key in self.val_metrics_sem.keys():
                y_pred_ = predictions['sem'] if 'ce' in key or 'acc' in key else pred_class
                if key.startswith('val_'):
                    self.val_metrics_sem[key](y_true=label_sem, y_pred=y_pred_, sample_weight=w_)
                elif info is not None:
                    key_ = key.replace('/','_').rsplit('_',1)[0]
                    if key_ in info:
                        info_bool = [key_ == x for x in info]
                        info_bool = np.array(info_bool).reshape(-1,1,1)

                        self.val_metrics_sem[key](y_true=label_sem, y_pred=y_pred_,
                                                sample_weight=w_*info_bool)
                
    def reset_sum_val(self):
        for key in self.val_metrics_reg.keys():
            self.val_metrics_reg[key].reset_states()
        for key in self.val_metrics_sem.keys():
            self.val_metrics_sem[key].reset_states()

    # @tf.function
    def summaries_val(self, step):
        with self.val_writer.as_default():
            if self.args.lambda_reg > 0.0:
                for key in self.val_metrics_reg.keys():
                    tf.summary.scalar(key.replace('_','/'),self.val_metrics_reg[key].result(),step=step)

            if self.args.lambda_reg < 1.0:    
                for key in self.val_metrics_sem.keys():
                    tf.summary.scalar(key.replace('_','/'),self.val_metrics_sem[key].result(),step=step)
        dict_reg = {k.split('_')[-1]:v.result().numpy() for k, v in self.val_metrics_reg.items()}
        dict_sem = {k.split('_')[-1]:v.result().numpy() for k, v in self.val_metrics_sem.items()}
        
        return {**dict_reg,**dict_sem}
            

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

    @tf_function()
    def forward_ntimes(self, x, is_training, n, return_moments=True):
        out_dict = {}
        out = [self.model(x,is_training) for _ in range(n)]

        for key in out[0].keys():
        # for key in ['reg','sem']: # not implemented for sem yet
            stacked = tf.stack([x[key] for x in out],axis=0)        
            if key in ['reg','sem']:
                if return_moments:
                    sum_x = tf.reduce_sum(stacked,axis=0)
                    sum_x2 = tf.reduce_sum(stacked**2, axis=0)
                    out_dict[key] = tf.stack((sum_x,sum_x2), axis=-1)
                else:
                    out_dict[key] = tf.reduce_mean(stacked,axis=0)
            else:
                out_dict[key] = tf.reduce_mean(stacked,axis=0)

        return out_dict

    # @tf_function()
    def forward_ensemble(self, x, is_training, ckpt_list, return_moments=True):
        out_dict = {}
        out = []
        for ckpt in ckpt_list:
            self.model.load_weights(ckpt)
            out.append(self.model(x,is_training))

        for key in out[0].keys():
        # for key in ['reg','sem']: # not implemented for sem yet
            stacked = tf.stack([x[key] for x in out],axis=0)        
            if key in ['reg','sem']:
                if return_moments:
                    sum_x = tf.reduce_sum(stacked,axis=0)
                    sum_x2 = tf.reduce_sum(stacked**2, axis=0)
                    out_dict[key] = tf.stack((sum_x,sum_x2), axis=-1)
                else:
                    out_dict[key] = tf.reduce_mean(stacked,axis=0)
            else:
                out_dict[key] = tf.reduce_mean(stacked,axis=0)

        return out_dict


    # def get_optimiter(self):
    #     if self.args.optimizer == 'adagrad':
    #         optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.args.lr)
    #     elif self.args.optimizer == 'adam':
    #         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.args.lr)
    #     elif self.args.optimizer == 'SGD':
    #         optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.args.lr)
    #     elif self.args.optimizer == 'momentum':
    #         optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.args.lr, momentum=0.9, use_nesterov=True)
    #     elif self.args.optimizer == 'annealing':
    #         learning_rate = tools.inv_lr_decay(self.args.lr, tf.compat.v1.train.get_global_step(), gamma=0.001, power=0.75)
    #         tf.compat.v1.summary.scalar('loss/annealing_lr', tf.math.log(learning_rate))
    #         optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
    #     else:
    #         raise ValueError('optimizer {} not defined'.format(self.args.optimizer))
    #     return optimizer
    def define_optimiter(self):
        # if self.args.optimizer == 'adagrad':
        #     optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.args.lr)
        if self.args.optimizer == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
        elif self.args.optimizer == 'sgd':
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=self.args.lr,
                decay_steps=self.args.lr_step,
                decay_rate=0.96, staircase=True,
                )
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=self.args.momentum)
        else:
            raise NotImplementedError


