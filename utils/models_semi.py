import tensorflow as tf
import numpy as np

import utils.tools_tf as tools

def bn_layer(X, activation_fn=None, is_training=True):
    if activation_fn is None: activation_fn = lambda x: x

    return activation_fn(tf.compat.v1.layers.batch_normalization(X, training=is_training))


def block(x, is_training, is_bn=True):
    x2 = tf.compat.v1.layers.conv2d(x, 64, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.compat.v1.layers.conv2d(x2, 64, kernel_size=3, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.compat.v1.layers.conv2d(x2, 256, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=None, is_training=is_training) if is_bn else x2

    return x2


def discriminator(input, scope_name='discriminator', is_training=True, is_bn=True, reuse=tf.compat.v1.AUTO_REUSE, return_feat = False):
    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x = tf.compat.v1.layers.conv2d(input, 64, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.compat.v1.layers.conv2d(x, 128, kernel_size=4, strides=2, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.compat.v1.layers.conv2d(x, 256, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        x1 = x
        x = tf.compat.v1.layers.conv2d(x, 512, kernel_size=4, strides=2, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.compat.v1.layers.conv2d(x, 2, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tools.bilinear(x, input.shape[1])
        if return_feat:
            return x, x1
        else:
            return x


def decode(input, scope_name='decode', is_training=True, is_bn=True, reuse=tf.compat.v1.AUTO_REUSE, scale=8, n_feat_last = None):
    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):
        n_feat = 128
        x = tf.compat.v1.layers.conv2d_transpose(input, 64, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        for i in range(int(np.log2(scale))):
            x = tf.compat.v1.layers.conv2d_transpose(x, 128, kernel_size=3, strides=2, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
            if n_feat_last is not None and i == int(np.log2(scale))-1:
                n_feat = n_feat_last
            x = tf.compat.v1.layers.conv2d_transpose(x, n_feat, kernel_size=3, strides=1, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x

def encode(input, scope_name='encode', is_training=True, is_bn=True, reuse=tf.compat.v1.AUTO_REUSE, scale=8):
    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):

        x = tf.compat.v1.layers.conv2d(input, 64, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        for _ in range(int(np.log2(scale))):
            x = tf.compat.v1.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

            x = tf.compat.v1.layers.conv2d(x, 128, kernel_size=3, strides=1, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x

def encode_same(input, scope_name='encode_same', is_training=True, is_bn=True, reuse=tf.compat.v1.AUTO_REUSE, is_small = True, dropout_always=True):
    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):

        x = tf.compat.v1.layers.conv2d(input, 64, 3, activation=tf.nn.relu, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        iters = 1 if is_small else 3

        for _ in range(iters):
            x = tf.compat.v1.layers.conv2d(x, 128, 3, activation=tf.nn.relu, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x
