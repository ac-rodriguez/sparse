import tensorflow as tf



from colorize import slice_last_dim
from tools_tf import bilinear, bn_layer, resid_block




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

def SR_task(feat_l, args, is_batch_norm=True, is_training=True):
    with tf.variable_scope('SR_task'):
        feat_l_up = bilinear(feat_l, size=args.patch_size * args.scale, name='LR_up')

        HR_hat1 = deep_sentinel2(feat_l_up, n_channels=3, is_residual=False, is_training=is_training,
                                 is_batch_norm=is_batch_norm)

        feat_l_rgb = slice_last_dim(feat_l_up, dims=(2, 1, 0))

        HR_hat1 = HR_hat1 + feat_l_rgb

        HR_hat = tf.layers.conv2d(HR_hat1, 3, 3, activation=tf.nn.sigmoid, padding='same')

        return HR_hat
