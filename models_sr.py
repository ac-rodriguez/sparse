import tensorflow as tf
import sys

from colorize import slice_last_dim
from tools_tf import bilinear, bn_layer, resid_block
import numpy as np

def deeplab(input, n_channels, is_training=True):
    with tf.variable_scope('resnet_blocks'):
        x = resid_block(input, filters=[128, 128], is_residual=False, is_training=is_training)
        for i in range(6):
            x = resid_block(x, is_training=is_training)

    hr_hat = tf.layers.conv2d(x, filters=n_channels, kernel_size=3, activation=None,
                              padding='same')
    return hr_hat


def deep_sentinel2(input, n_channels, is_residual=True, scope_name='resnet_blocks', is_training=True,
                   is_batch_norm=True):
    feature_size = 128
    with tf.variable_scope(scope_name):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x = tf.layers.conv2d(input, feature_size, kernel_size=3, activation=tf.nn.relu, padding='same')
        for i in range(6):
            # features_nn = resid_block(features_nn)
            x = resid_block(x, filters=[feature_size, feature_size], is_residual=True, is_training=is_training,
                            is_batch_norm=is_batch_norm)
            # features_nn = resid_block(features_nn)

    hr_hat = tf.layers.conv2d(x, filters=n_channels, kernel_size=3, activation=None,
                              padding='same')
    if is_residual:
        return hr_hat + input
    else:
        return hr_hat


def SR_task_old(feat_l, args, is_batch_norm=True, is_training=True):
    with tf.variable_scope('SR_task'):
        feat_l_up = bilinear(feat_l, size=args.patch_size * args.scale, name='LR_up')

        HR_hat1 = deep_sentinel2(feat_l_up, n_channels=3, is_residual=False, is_training=is_training,
                                 is_batch_norm=is_batch_norm)

        feat_l_rgb = slice_last_dim(feat_l_up, dims=(2, 1, 0))

        HR_hat1 = tf.nn.sigmoid(HR_hat1) + bn_layer(feat_l_rgb, is_training=is_training)

        HR_hat = tf.layers.conv2d(HR_hat1, 3, 3, activation=tf.nn.sigmoid, padding='same')

        return HR_hat
def SR_task(feat_l, size, is_batch_norm=True, is_training=True):
    with tf.variable_scope('SR_task'):
        feat_l_up = bilinear(feat_l, size=size, name='LR_up')

        HR_hat1 = deep_sentinel2(feat_l_up, n_channels=3, is_residual=False, is_training=is_training,
                                 is_batch_norm=is_batch_norm)

        feat_l_rgb = slice_last_dim(feat_l_up, dims=(2, 1, 0))

        feat_l_rgb = tf.layers.conv2d(feat_l_rgb, 3, 3, activation=None, padding='same')
        feat_l_rgb = tf.layers.conv2d(feat_l_rgb, 3, 1, activation=tf.nn.sigmoid, padding='same')

        HR_hat1 = tf.nn.sigmoid(HR_hat1) + bn_layer(feat_l_rgb, is_training=is_training)

        HR_hat = tf.layers.conv2d(HR_hat1, 3, 3, activation=None, padding='same')
        HR_hat = tf.layers.conv2d(HR_hat, 3, 1, activation=tf.nn.sigmoid, padding='same')

        return HR_hat

def SR_task1(feat_l, size, is_batch_norm=True, is_training=True):
    with tf.variable_scope('SR_task'):
        feat_l_up = bilinear(feat_l, size=size, name='LR_up')

        HR_hat1 = deep_sentinel2(feat_l_up, n_channels=3, is_residual=False, is_training=is_training,
                                 is_batch_norm=is_batch_norm)

        feat_l_rgb = slice_last_dim(feat_l_up, dims=(2, 1, 0))

        HR_hat1 = HR_hat1 + feat_l_rgb

        HR_hat = tf.layers.conv2d(HR_hat1, 3, 3, activation=tf.nn.sigmoid, padding='same')

        return HR_hat


def ConvBlock(X, filters, kernel, stride, padding, bias=True, activation_fn=None, is_bn=True, is_training=True):
    out = tf.layers.conv2d(X, filters=filters, kernel_size=kernel, strides=stride, padding=padding, use_bias=bias)
    if is_bn:
        out = bn_layer(out, is_training=is_training)
    if activation_fn is None: activation_fn = lambda x: x

    return activation_fn(out)


def DeconvBlock(X, filters, kernel, stride, padding, bias=True,activation_fn=None, is_bn=True, is_training=True):
    X = tf.layers.conv2d_transpose(X, filters=filters, kernel_size=kernel, strides=stride, padding=padding, use_bias=bias)
    if is_bn:
        X = bn_layer(X, is_training=is_training)
    if activation_fn is None: activation_fn = lambda x: x

    return activation_fn(X)


def UpBlock(X, filters, kernel, stride, padding, bias=True, activation_fn=None, is_training=True):
    h0 = DeconvBlock(X,filters=filters,kernel=kernel,stride=stride,padding=padding,bias=bias,activation_fn=activation_fn,is_training=is_training)
    l0 = ConvBlock(h0, filters=filters, kernel=kernel, stride=stride, padding=padding, bias=bias,
                     activation_fn=activation_fn, is_training=is_training)
    h1 = DeconvBlock(l0 - X,filters=filters,kernel=kernel,stride=stride,padding=padding,bias=bias,activation_fn=activation_fn,is_training=is_training)

    return h1+ h0
def DownBlock(X, filters, kernel, stride, padding, bias=True, activation_fn=None, is_training=True):
    l0 = ConvBlock(X,filters=filters,kernel=kernel,stride=stride,padding=padding,bias=bias,activation_fn=activation_fn,is_training=is_training)
    h0 = DeconvBlock(l0, filters=filters, kernel=kernel, stride=stride, padding=padding, bias=bias,
                     activation_fn=activation_fn, is_training=is_training)
    l1 = ConvBlock(h0 - X,filters=filters,kernel=kernel,stride=stride,padding=padding,bias=bias,activation_fn=activation_fn,is_training=is_training)

    return l1+ l0


def prelu(_x):
    # with tf.variable_scope('prelu',reuse=False):
    #     alphas = tf.get_variable('alpha', _x.get_shape()[-1],
    #                              initializer=tf.constant_initializer(0.0),
    #                              dtype=tf.float32)
    #
    #     pos = tf.nn.relu(_x)
    #     neg = alphas * (_x - abs(_x)) * 0.5
    #
    #     return pos + neg
    ## TODO
    # return tf.keras.layers.PReLU(_x)
    return tf.nn.relu(_x)


def dbpn_SR(feat_l,scale=2, is_training=True, deep=1):

    if scale == 2:
        kernel = 6
        stride = 2
    elif scale == 4:
        kernel = 8
        stride = 4
    elif scale == 8:
        kernel = 12
        stride = 8
    else:
        print('scale {} not defined'.format(scale))
        sys.exit(1)

    feat = 256
    base_filter = 64

    X = ConvBlock(feat_l, filters=feat, kernel=kernel, stride=1, padding='SAME', activation_fn=prelu, is_training=is_training)
    X = ConvBlock(X, filters=base_filter, kernel=1, stride=1, padding='VALID', activation_fn=prelu, is_training=is_training)

    h1 = UpBlock(X, filters=base_filter,kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)
    l1 = DownBlock(h1, filters=base_filter,kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)
    h2 = UpBlock(l1, filters=base_filter,kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)

    concat_h = tf.concat([h1,h2],axis=3)
    l = DownBlock(concat_h, filters=base_filter*2,kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)

    concat_l = tf.concat([l, l1],axis=3)
    h = UpBlock(concat_l, filters=base_filter*3,kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)

    for _ in range(deep):
        concat_h = tf.concat([h, concat_h],axis=3)
        l = DownBlock(concat_h, filters=concat_h.shape[-1], kernel=kernel, stride=stride, padding='SAME',
                      activation_fn=prelu, is_training=is_training)

        concat_l = tf.concat([l, concat_l], axis=3)
        h = UpBlock(concat_l, filters=concat_l.shape[-1], kernel=kernel, stride=stride, padding='SAME',
                    activation_fn=prelu, is_training=is_training)

    concat_h = tf.concat([h, concat_h],axis=3)

    HR_hat = tf.layers.conv2d(concat_h, 3, 3, activation=tf.nn.sigmoid, padding='same')

    return HR_hat


def dbpn_LR(x,scale=2, is_training=True):

    if scale < 8:
        kernel = 3
    else:
        kernel = 4
    stride=2
    base_filter = 256

    l = ConvBlock(x, filters=base_filter, kernel=kernel, stride=1, padding='SAME', activation_fn=prelu, is_training=is_training)
    # l = DownBlock(X, filters=base_filter, kernel=kernel, stride=stride, padding='SAME', activation_fn=prelu,
    #                is_training=is_training)

    for i in range(int(np.log2(scale))):

        # h1 = UpBlock(l, filters=base_filter*(i+1),kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)
        l = DownBlock(l, filters=base_filter,kernel=kernel,stride=stride,padding='SAME', activation_fn=prelu, is_training=is_training)
        # l = tf.concat([l, l2], axis=3)

    return tf.nn.relu(l)

