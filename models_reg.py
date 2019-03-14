
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



def countception(input,pad, scope_name='countception', is_training=True, is_return_feat=False, reuse=tf.AUTO_REUSE):

    def selu(x):
        with tf.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.where(x >= 0., x, alpha * tf.nn.elu(x))


    def ConvLayer(x, num_filters, kernel_size, name, pad='SAME', is_last=False):
        with tf.variable_scope(name):
            w = tf.get_variable('weights', shape=[kernel_size[0], kernel_size[1],
                                                  x.get_shape()[3], num_filters],
                                initializer=tf.contrib.layers.xavier_initializer())
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=pad)
            if is_last:
                b = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
                return conv + b
            else:
                # b = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
                bn = tf.layers.batch_normalization(conv, training=(is_training))
                return tf.nn.relu(bn)
                # return selu(conv + b)


    def ConcatBlock(x, num_filters1, num_filters2, name):
        with tf.variable_scope(name):
            conv1x1 = ConvLayer(x, num_filters1, [1, 1], 'conv1x1', pad='VALID')
            conv3x3 = ConvLayer(x, num_filters2, [3, 3], 'conv3x3', pad='SAME')
            return tf.concat([conv1x1, conv3x3], axis=-1)

    with tf.variable_scope(scope_name, reuse=reuse):
        # pad = 32
        net = tf.pad(input, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'CONSTANT')
        # net = input
        net = ConvLayer(net, 64, [3, 3], name='conv1', pad='VALID')
        net = ConcatBlock(net, 16, 16, name='concat_block1')
        net = ConcatBlock(net, 16, 32, name='concat_block2')
        net = ConvLayer(net, 16, [14, 14], name='conv2', pad='VALID')
        net = ConcatBlock(net, 112, 48, name='concat_block3')
        net = ConcatBlock(net, 64, 32, name='concat_block4')
        net = ConcatBlock(net, 40, 40, name='concat_block5')
        net1 = net
        net = ConcatBlock(net, 32, 96, name='concat_block6')
        net = ConvLayer(net, 32, [17, 17], name='conv3', pad='VALID')
        net = ConvLayer(net, 64, [1, 1], name='conv4', pad='VALID')
        net = ConvLayer(net, 64, [1, 1], name='conv5', pad='VALID')
        net_reg = ConvLayer(net, 1, [1, 1], name='out_reg', pad='VALID', is_last=True)
        net_sem = ConvLayer(net, 2, [1, 1], name='out_sem', pad='VALID', is_last=True)

        net_reg = tf.image.crop_to_bounding_box(net_reg, 0, 0, input.shape[1],input.shape[2])
        net_sem = tf.image.crop_to_bounding_box(net_sem, 0, 0, input.shape[1],input.shape[2])
    if is_return_feat:
        return {'reg': net_reg, 'sem': net_sem}, net1
    else:
        return {'reg': net_reg, 'sem': net_sem}

