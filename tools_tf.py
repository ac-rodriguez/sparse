import tensorflow as tf
from tensorflow.python.ops import math_ops


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


def avg_pool(X, scale, name=None):
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
