import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np

def bn_layer(X, activation_fn=None, is_training=True):
    if activation_fn is None: activation_fn = lambda x: x
    return activation_fn(tf.layers.batch_normalization(X, training=is_training))


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


def resid_block1(X, filters=[64, 128], is_residual=False, scale=0.1):
    Xr = tf.layers.conv2d(X, filters=filters[0], kernel_size=3, activation=tf.nn.relu, padding='same')
    # Xr = bn_layer(Xr, tf.nn.relu)
    Xr = tf.layers.conv2d(Xr, filters=filters[1], kernel_size=1, activation=tf.nn.relu, padding='same')

    Xr = Xr * scale

    if is_residual:
        return X + Xr
    else:
        return Xr


def sum_pool(X, scale, name=None):
    return tf.multiply(float(scale ** 2),
                       tf.nn.avg_pool(X, ksize=(1, scale, scale, 1),
                                      strides=(1, scale, scale, 1), padding='VALID'),
                       name=name)

def max_pool(X, scale, name=None):
    return tf.nn.max_pool(X, ksize=(1, scale, scale, 1),
                                      strides=(1, scale, scale, 1), padding='VALID', name=name)

def avg_pool(X, scale, name=None):
    if len(X.shape) == 3: X = tf.expand_dims(X,-1)
    return tf.nn.avg_pool(X, ksize=(1, scale, scale, 1),
                          strides=(1, scale, scale, 1), padding='VALID', name=name)


def bilinear(X, size, name=None):
    return tf.image.resize_bilinear(X, size=[int(size), int(size)], name=name)


log10 = lambda x: tf.log(x) / tf.log(10.0)


def s2n(a, b):
    sn = tf.reduce_mean(tf.squared_difference(a, b))
    sn = 10 * log10(255.0 / sn)

    return sn


def snr_metric(a, b):
    sd, sd_op = tf.metrics.mean_squared_error(a, b)

    s2n = 10 * log10(255.0 / sd)

    return s2n, sd_op


def get_lr_ADAM(optimizer, learning_rate):
    beta1_power, beta2_power = optimizer._get_beta_accumulators()
    optim_learning_rate = (learning_rate * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

    return optim_learning_rate



class SessionHook(tf.train.SessionRunHook):

    def __init__(self, values, labels, size=1):
        super(SessionHook, self).__init__()

        # self.iterator_initializer_func = None

        self._tensors = None

        # values = scopes['net_scope'] + '/' + scopes['emb_scope'] + '/' + scopes['emb_name'] + '/BiasAdd:0'
        # labels = scopes['metrics_scope'] + '/' + scopes['label_name'] + ':0'

        self._tensor_names = [values, labels]
        self._embeddings = [[], []]
        self.size= size
        self.iter = 0

    def begin(self):
        self._tensors = [tf.get_default_graph().get_tensor_by_name(x) for x in self._tensor_names]

    def after_create_session(self, session, coord):
        self._embeddings = [[], []]

        pass
    #     """ Initialise the iterator after the session has been created."""
    #     self.iterator_initializer_func(session)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._tensors)

    def after_run(self, run_context, run_values):
        if self.iter == 0:
            self.n_batch = run_values[0][0].shape[0]
            self.n_feat = run_values[0][0].shape[-1]
            self.n_pixels = run_values[0][0].shape[1] * run_values[0][0].shape[2]
            self.sample_ind = np.random.choice(self.n_pixels,int(self.n_pixels *0.1), replace=False)
        if self.iter <= self.size:
            val_ = run_values[0][0].reshape(self.n_batch, -1,self.n_feat)[:,self.sample_ind,:].reshape(-1,self.n_feat)
            lab_ = run_values[0][1].reshape(self.n_batch, -1)[:,self.sample_ind].reshape(-1)
            # lab_ = np.mean(lab_,axis=0)
            self._embeddings[0].extend(val_)
            self._embeddings[1].extend(lab_)
            self.iter+=1

    def end(self, session):
        pass

    def get_embeddings(self):
        return {
            'values': self._embeddings[0],
            'labels': self._embeddings[1],
        }

    # def set_iterator_initializer(self, fun):
    #     self.iterator_initializer_func = fun
