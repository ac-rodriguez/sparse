
import tensorflow as tf

i = 0
# from tools_tf import bn_layer

def bn_layer(X, activation_fn=None, is_training=True):
    if activation_fn is None: activation_fn = lambda x: x
    global i

    with tf.variable_scope('batch_norm'+str(i), reuse=False):
        out =  activation_fn(tf.layers.batch_normalization(X, training=is_training))

        gamma = tf.trainable_variables(tf.get_variable_scope().name)[0]
        beta = tf.trainable_variables(tf.get_variable_scope().name)[1]
        tf.summary.histogram('bn/gamma',gamma)
        tf.summary.histogram('bn/beta',beta)
        i+=1
        return out

def block(x, is_training, is_bn = True):
    x2 = tf.layers.conv2d(x, 64, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.layers.conv2d(x2, 64, kernel_size=3, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.layers.conv2d(x2, 256, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=None, is_training=is_training) if is_bn else x2

    return x2


def simple(input, n_channels, scope_name='simple', is_training=True, is_bn=True):

    feature_size = 256

    global i
    i = 0
    with tf.variable_scope(scope_name):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x1 = tf.layers.conv2d(input, feature_size, kernel_size=3, use_bias=False, activation=tf.nn.relu, padding='same')
        x1bn = bn_layer(x1, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x1
        x1bn1 = tf.layers.conv2d(x1bn, feature_size, kernel_size=1, use_bias=False, padding='same')
        x1bn1 = bn_layer(x1bn1, is_training=is_training) if is_bn else x1bn1

        x2 = block(x1bn, is_training, is_bn)

        x3_ = tf.nn.relu(x1bn1 + x2)
        x3 = block(x3_, is_training, is_bn)

        x4_ = tf.nn.relu(x3_ + x3)
        x4 = block(x4_, is_training, is_bn)

        x5_ = tf.nn.relu(x4_ + x4)
        x5 = block(x5_, is_training, is_bn)

        x6_ = tf.nn.relu(x5_ + x5)
        x6 = block(x6_, is_training, is_bn)

        x7_ = tf.nn.relu(x6_ + x6)
        x7 = block(x7_, is_training, is_bn)

        # Regression
        x8a_ = tf.nn.relu(x7_ + x7)
        x8a = tf.layers.conv2d(x8a_, n_channels, kernel_size=3, use_bias=False, padding='same')

        # Semantic
        x8b_ = tf.nn.relu(x7_ + x7)
        x8b = tf.layers.conv2d(x8b_, 2, kernel_size=3, use_bias=False, padding='same')

    return {'reg': x8a, 'sem': x8b}
    # return x8a
