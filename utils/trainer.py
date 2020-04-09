import tensorflow as tf
import numpy as np
cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy

# from AdaBound import AdaBoundOptimizer

from utils.colorize import colorize, inv_preprocess_tf
# from utils.models_reg import simple, countception

from utils.models_reg import SimpleA
import utils.tools_tf as tools
# import utils.models_semi as semi

def tf_function():
    def decorator(func):
        if tf.__version__.startswith('2.'):
            return tf.function(func)
        else:
            return func
    return decorator

class Trainer():

    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        self.model_dir = args.model_dir

        self.scale = 1
        self.is_slim = self.args.is_slim_eval

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
            self.model = SimpleA(self.n_classes,extra_depth=depth)
        else:
            raise NotImplementedError

        self.mse_loss = tf.keras.losses.MeanSquaredError()
        self.ce_loss = tf.keras.losses.SparseCategoricalCrossentropy()


        self.train_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_ce = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.train_mse = tf.keras.metrics.MeanSquaredError()
        self.train_mae = tf.keras.metrics.MeanAbsoluteError()

        self.val_mse = tf.keras.metrics.MeanSquaredError()
        self.val_mae = tf.keras.metrics.MeanAbsoluteError()

        self.val_ce = tf.keras.metrics.SparseCategoricalCrossentropy()
        self.val_acc = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_iou = tf.keras.metrics.MeanIoU(self.n_classes+1)
        self.val_prec = tf.keras.metrics.Precision()
        self.val_rec = tf.keras.metrics.Recall()
        # add prec and recall
        self.define_optimiter()

        # Define summary locs:

        self.train_writer = tf.summary.create_file_writer(self.model_dir)
        self.val_writer = tf.summary.create_file_writer(self.model_dir+'/val')

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
            predictions = self.model(features['feat_l'], is_training=True)
            loss = self.compute_loss(predictions, labels)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return predictions

    # @tf.function
    def test_step(self, features, labels):
        predictions = self.model(features['feat_l'], is_training=False)
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
                tf.summary.scalar('train/mse',self.train_mse.result(),step=step)
                tf.summary.scalar('train/mae',self.train_mae.result(),step=step)
            if self.args.lambda_reg < 1.0:
                tf.summary.scalar('train/ce',self.train_ce.result(),step=step)
    
    def reset_sum_train(self):
        self.train_mse.reset_states()
        self.train_mae.reset_states()
        self.train_ce.reset_states()

    def update_sum_val(self, predictions, labels):
        
        label_sem, w = self.get_sem(labels, return_w=True)

        if self.args.lambda_reg > 0.0:
            w_reg = self.get_w(labels, is_99=True)
            self.val_mse(y_true=labels, y_pred=predictions['reg'],
                                                              sample_weight=w_reg)
            self.val_mae(y_true=labels, y_pred=predictions['reg'],
                                                              sample_weight=w_reg)

        if self.args.lambda_reg < 1.0:
            w_ = self.float_(tf.reduce_any(tf.greater(w, 0), -1))
            self.val_ce(y_true=label_sem, y_pred=predictions['sem'], sample_weight=w_)
            self.val_acc(y_true=label_sem,y_pred=predictions['sem'], sample_weight=w_)

            pred_class = tf.argmax(input=predictions['sem'], axis=3)

            self.val_iou(y_true=label_sem,y_pred=pred_class, sample_weight=w_)
            self.val_prec(y_true=label_sem,y_pred=pred_class, sample_weight=w_)
            self.val_rec(y_true=label_sem,y_pred=pred_class, sample_weight=w_)


    def reset_sum_val(self):
        self.val_mse.reset_states()
        self.val_mae.reset_states()
        
        self.val_ce.reset_states()            
        self.val_acc.reset_states()        

        self.val_iou.reset_states()        
        self.val_prec.reset_states()        
        self.val_rec.reset_states()        

    # @tf.function
    def summaries_val(self, step):
        with self.val_writer.as_default():
            if self.args.lambda_reg > 0.0:
                tf.summary.scalar('val/mse',self.val_mse.result(),step=step)
                tf.summary.scalar('val/mae',self.val_mse.result(),step=step)
            if self.args.lambda_reg < 1.0:    
                tf.summary.scalar('val/ce',self.val_ce.result(),step=step)
                tf.summary.scalar('val/acc',self.val_acc.result(),step=step)

                tf.summary.scalar('val/iou',self.val_iou.result(),step=step)
                tf.summary.scalar('val/precision',self.val_prec.result(),step=step)
                tf.summary.scalar('val/recall',self.val_rec.result(),step=step)
        
        return {'mse':self.val_mse.result().numpy(),
                'mae':self.val_mae.result().numpy(),
                'ce':self.val_ce.result().numpy(),
                'acc':self.val_acc.result().numpy(),
                'iou':self.val_iou.result().numpy(),
                'precision':self.val_prec.result().numpy(),
                'recall':self.val_rec.result().numpy()
                }
            

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
        else:
            raise NotImplementedError


