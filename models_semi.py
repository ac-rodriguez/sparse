import tensorflow as tf
import tools_tf as tools
import numpy as np

def bn_layer(X, activation_fn=None, is_training=True):
    if activation_fn is None: activation_fn = lambda x: x

    return activation_fn(tf.layers.batch_normalization(X, training=is_training))


def block(x, is_training, is_bn=True):
    x2 = tf.layers.conv2d(x, 64, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.layers.conv2d(x2, 64, kernel_size=3, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.layers.conv2d(x2, 256, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=None, is_training=is_training) if is_bn else x2

    return x2


def discriminator(input, scope_name='discriminator', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, return_feat = False):
    with tf.variable_scope(scope_name, reuse=reuse):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x = tf.layers.conv2d(input, 64, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 128, kernel_size=4, strides=2, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 256, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        x1 = x
        x = tf.layers.conv2d(x, 512, kernel_size=4, strides=2, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 2, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tools.bilinear(x, input.shape[1])
        if return_feat:
            return x, x1
        else:
            return x


def encode_HR(input, scope_name='encode_HR', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, scale=8):
    with tf.variable_scope(scope_name, reuse=reuse):

        x = tf.layers.conv2d(input, 64, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        for _ in range(int(np.log2(scale))):
            x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

            x = tf.layers.conv2d(x, 256, kernel_size=3, strides=1, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x
