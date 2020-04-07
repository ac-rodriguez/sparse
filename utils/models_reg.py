import sys
import tensorflow as tf

i = 0
# from tools_tf import bn_layer




from tensorflow.keras import layers



class Block(layers.Layer):
    def __init__(self, nfeatures = [64,64,256]):
        super(Block,self).__init__()
        self.conv1 = layers.Conv2D(nfeatures[0], 3, activation='relu', padding='same')
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(nfeatures[1], 3, activation='relu', padding='same')
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(nfeatures[2], 3, activation='relu', padding='same')
        self.bn3 = layers.BatchNormalization()


    def call(self,x,is_training):
        x = self.conv1(x)
        x = self.bn1(x,is_training)

        x = self.conv2(x)
        x = self.bn2(x,is_training)

        x = self.conv3(x)
        x = self.bn3(x,is_training)
        return x



class SimpleA(tf.keras.Model):
  def __init__(self,n_classes, extra_depth=0, lambda_reg=0.5):
    super(SimpleA, self).__init__()

    self.extra_depth = extra_depth
    self.n_classes = n_classes
    self.lambda_reg = lambda_reg

    # Encode same part
    self.block0 = Block(nfeatures=[64,18,128])

    # self.conv1 = layers.Conv2D(64,3,activation='relu',padding='same')
    # self.bn1 = layers.BatchNormalization()
    #
    # self.conv2 = layers.Conv2D(128,3,activation='relu',padding='same')
    # self.bn2 = layers.BatchNormalization()
    #
    # self.conv3 = layers.Conv2D(128,3,activation='relu',padding='same')
    # self.bn3 = layers.BatchNormalization()

    self.conv4 = layers.Conv2D(128,3,activation='relu',padding='same')
    self.bn4 = layers.BatchNormalization()

    feature_size = 256

    self.conv5 = layers.Conv2D(feature_size,3, activation='relu',padding='same', use_bias=False)
    self.bn5 = layers.BatchNormalization()

    self.conv5a = layers.Conv2D(feature_size,1, activation='relu',padding='same', use_bias=False)
    self.bn5a = layers.BatchNormalization()

    self.block1 = Block()
    self.block2 = Block()
    self.block3 = Block()
    self.block4 = Block()
    self.block5 = Block()
    self.block6 = Block()

    self.extra_blocks = [Block() for _ in range(self.extra_depth)]

    self.relu =layers.ReLU()

    if self.lambda_reg > 0.0:
        self.conv_reg = layers.Conv2D(n_classes,3, activation=None,padding='same', use_bias=False)
    if self.lambda_reg < 1.0:
        self.conv_sem = layers.Conv2D(n_classes+1, 3, activation=None, padding='same', use_bias=False)


  def call(self, x, is_training, return_feat=False):

    x = self.block0(x, is_training)
    # x = self.bn1(self.conv1(x))
    # x = self.bn2(self.conv2(x))
    # x = self.bn3(self.conv3(x))
    x = self.bn4(self.conv4(x))

    x = self.bn5(self.conv5(x))
    x1 = self.bn5a(self.conv5a(x))

    x2 = self.block1(x, is_training)

    x3_ = self.relu(x1 + x2)
    x3 = self.block2(x3_, is_training)

    x4_ = self.relu(x3_ + x3)
    x4 = self.block3(x4_, is_training)

    x5_ = self.relu(x4_ + x4)
    x5 = self.block4(x5_, is_training)
    mid = x5_
    x6_ = self.relu(x5_ + x5)
    x6 = self.block5(x6_, is_training)

    x7_ = self.relu(x6_ + x6)
    x7 = self.block6(x7_, is_training)

    for b in self.extra_blocks:

        x7_ = self.relu(x7_ + x7)
        x7 = b(x7_, is_training)

    last = self.relu(x7_ + x7)

    return_dict = {}
    if self.lambda_reg > 0.0:
        return_dict['reg'] = self.conv_reg(last)
    if self.lambda_reg < 1.0:
        return_dict['sem'] = self.conv_sem(last)
    
    if return_feat:
        raise NotImplementedError
        return {'reg': x_reg, 'sem': x_sem}, mid, last
    else:
        return return_dict


def bn_layer(X, activation_fn=None, is_training=True):
    if activation_fn is None: activation_fn = lambda x: x
    global i

    with tf.compat.v1.variable_scope('batch_norm'+str(i), reuse=False):
        out =  activation_fn(tf.compat.v1.layers.batch_normalization(X, training=is_training))

        gamma = tf.compat.v1.trainable_variables(tf.compat.v1.get_variable_scope().name)[0]
        beta = tf.compat.v1.trainable_variables(tf.compat.v1.get_variable_scope().name)[1]
        tf.compat.v1.summary.histogram('bn/gamma',gamma)
        tf.compat.v1.summary.histogram('bn/beta',beta)
        i+=1
        return out

def block(x, is_training, is_bn = True):
    x2 = tf.compat.v1.layers.conv2d(x, 64, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.compat.v1.layers.conv2d(x2, 64, kernel_size=3, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x2
    x2 = tf.compat.v1.layers.conv2d(x2, 256, kernel_size=1, use_bias=False, padding='same')
    x2 = bn_layer(x2, activation_fn=None, is_training=is_training) if is_bn else x2

    return x2


def simple(input, n_classes, scope_name='simple', is_training=True, is_bn=True, reuse=tf.compat.v1.AUTO_REUSE,
           return_feat=False, deeper=None):

    feature_size = 256

    global i
    i = 0
    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):
        # features_nn = resid_block(A_cube, filters=[128, 128], only_resid=True)
        x1 = tf.compat.v1.layers.conv2d(input, feature_size, kernel_size=3, use_bias=False, activation=tf.nn.relu, padding='same')
        x1bn = bn_layer(x1, activation_fn=tf.nn.relu, is_training=is_training) if is_bn else x1
        x1bn1 = tf.compat.v1.layers.conv2d(x1bn, feature_size, kernel_size=1, use_bias=False, padding='same')
        x1bn1 = bn_layer(x1bn1, is_training=is_training) if is_bn else x1bn1

        x2 = block(x1bn, is_training, is_bn)

        x3_ = tf.nn.relu(x1bn1 + x2)
        x3 = block(x3_, is_training, is_bn)

        x4_ = tf.nn.relu(x3_ + x3)
        x4 = block(x4_, is_training, is_bn)

        x5_ = tf.nn.relu(x4_ + x4)
        x5 = block(x5_, is_training, is_bn)
        mid = x5_
        x6_ = tf.nn.relu(x5_ + x5)
        x6 = block(x6_, is_training, is_bn)

        x7_ = tf.nn.relu(x6_ + x6)
        x7 = block(x7_, is_training, is_bn)

        if deeper is not None:
            for _ in range(deeper):
                x7_ = tf.nn.relu(x7_ + x7)
                x7 = block(x7_, is_training, is_bn)

        # Regression
        last = tf.nn.relu(x7_ + x7)
        # last = x8a_
        x8a = tf.compat.v1.layers.conv2d(last, n_classes, kernel_size=3, use_bias=False, padding='same')

        # Semantic
        # x8b_ = tf.nn.relu(x7_ + x7)
        x8b = tf.compat.v1.layers.conv2d(last, n_classes + 1, kernel_size=3, use_bias=False, padding='same')

    if return_feat:
        return {'reg': x8a, 'sem': x8b}, mid, last
    else:
        return {'reg': x8a, 'sem': x8b}
    # return x8a

# from tensorflow.contrib.slim.nets import resnet_v2
# from tensorflow.contrib import layers as layers_lib
# from tensorflow.contrib.framework.python.ops import arg_scope
# from tensorflow.contrib.layers.python.layers import layers
# from tensorflow.contrib.slim.python.slim.nets import resnet_utils

# def dl3(inputs, n_channels, is_training, base_architecture='resnet_v2_50', return_feat=False):

#     output_stride = 8

#     inputs_size = inputs.shape[1:3]
#     if base_architecture == 'resnet_v2_50':
#         base_model = resnet_v2.resnet_v2_50
#     else:
#         base_model = resnet_v2.resnet_v2_101


#     with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
#         _, end_points = base_model(inputs,
#                                    num_classes=n_channels,
#                                    is_training=is_training,
#                                    global_pool=False,
#                                    output_stride=output_stride)

#     mid_feat_name = [x for x in end_points if x.endswith(base_architecture + '/block4')][0]
#     mid_feat = end_points[mid_feat_name]

#     last = atrous_spatial_pyramid_pooling(mid_feat, output_stride, is_training)


#     with tf.compat.v1.variable_scope("upsampling_logits"):
#         x_log = layers_lib.conv2d(last, n_channels, [1, 1], activation_fn=None, normalizer_fn=None,
#                                 scope='conv_1x1')
#         logits = tf.image.resize_bilinear(x_log, inputs_size, name='upsample')

#     # net = end_points[base_architecture + '/block4']
#     # mid_feat2 = atrous_spatial_pyramid_pooling(mid_feat, output_stride, batch_norm_decay, is_training, scope='aspp_reg')
#     with tf.compat.v1.variable_scope("upsampling_reg"):
#         x_reg = layers_lib.conv2d(last, 1, [1, 1], activation_fn=None, normalizer_fn=None,
#                                 scope='conv_1x1')
#         pred_reg = tf.image.resize_bilinear(x_reg, inputs_size, name='upsample')

#     if return_feat:
#         return {'reg': pred_reg, 'sem': logits},  mid_feat, last
#     else:
#         return {'reg': pred_reg, 'sem': logits}


# def atrous_spatial_pyramid_pooling(inputs, output_stride, is_training, depth=256, scope = 'aspp'):
#   """Atrous Spatial Pyramid Pooling.

#   Args:
#     inputs: A tensor of size [batch, height, width, channels].
#     output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
#       the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
#     batch_norm_decay: The moving average decay when estimating layer activation
#       statistics in batch normalization.
#     is_training: A boolean denoting whether the input is for training.
#     depth: The depth of the ResNet unit output.

#   Returns:
#     The atrous spatial pyramid pooling output.
#   """
#   with tf.compat.v1.variable_scope(scope):
#     if output_stride not in [8, 16]:
#       raise ValueError('output_stride must be either 8 or 16.')

#     atrous_rates = [6, 12, 18]
#     if output_stride == 8:
#       atrous_rates = [2*rate for rate in atrous_rates]

#     with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
#       with arg_scope([layers.batch_norm], is_training=is_training):
#         inputs_size = tf.shape(input=inputs)[1:3]
#         # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
#         # the rates are doubled when output stride = 8.
#         conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
#         conv_3x3_1 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[0], scope='conv_3x3_1')
#         conv_3x3_2 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[1], scope='conv_3x3_2')
#         conv_3x3_3 = resnet_utils.conv2d_same(inputs, depth, 3, stride=1, rate=atrous_rates[2], scope='conv_3x3_3')

#         # (b) the image-level features
#         with tf.compat.v1.variable_scope("image_level_features"):
#           # global average pooling
#           image_level_features = tf.reduce_mean(input_tensor=inputs, axis=[1, 2], name='global_average_pooling', keepdims=True)
#           # 1x1 convolution with 256 filters( and batch normalization)
#           image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1, scope='conv_1x1')
#           # bilinearly upsample features
#           image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size,name='upsample')

#         net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3, name='concat')
#         net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')

#         return net




def countception(input,pad, scope_name='countception', is_training=True, is_return_feat=False, reuse=tf.compat.v1.AUTO_REUSE, config_volume=None):

    def selu(x):
        with tf.compat.v1.name_scope('elu') as scope:
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale * tf.compat.v1.where(x >= 0., x, alpha * tf.nn.elu(x))


    def ConvLayer(x, num_filters, kernel_size, name, pad='SAME', is_last=False):
        with tf.compat.v1.variable_scope(name):
            w = tf.compat.v1.get_variable('weights', shape=[kernel_size[0], kernel_size[1],
                                                  x.get_shape()[3], num_filters],
                                initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"))
            conv = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding=pad)
            if is_last:
                b = tf.compat.v1.get_variable('biases', [num_filters], initializer=tf.compat.v1.zeros_initializer())
                return conv + b
            else:
                # b = tf.get_variable('biases', [num_filters], initializer=tf.zeros_initializer())
                bn = tf.compat.v1.layers.batch_normalization(conv, training=(is_training))
                return tf.nn.relu(bn)
                # return selu(conv + b)


    def ConcatBlock(x, num_filters1, num_filters2, name):
        with tf.compat.v1.variable_scope(name):
            conv1x1 = ConvLayer(x, num_filters1, [1, 1], 'conv1x1', pad='VALID')
            conv3x3 = ConvLayer(x, num_filters2, [3, 3], 'conv3x3', pad='SAME')
            return tf.concat([conv1x1, conv3x3], axis=-1)

    n_filters_last = 64
    n_filters_mid = 40
    if config_volume is not None:

        if config_volume['last']:
            if config_volume['hr']:
                n_filters_last = 3
            else:
                n_filters_last = 3*(config_volume['scale']**2)
        else:
            if config_volume['hr']:
                n_filters_mid = 3
            else:
                if config_volume['scale'] == 8:
                    n_filters_mid = 3*20
                elif config_volume['scale'] == 16:
                    n_filters_mid = 3 * 72
                else:
                    print('not implemented error')
                    sys.exit(1)

    with tf.compat.v1.variable_scope(scope_name, reuse=reuse):
        # pad = 32
        net = tf.pad(tensor=input, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='CONSTANT')
        # net = input
        net = ConvLayer(net, 64, [3, 3], name='conv1', pad='VALID')
        net = ConcatBlock(net, 16, 16, name='concat_block1')
        net = ConcatBlock(net, 16, 32, name='concat_block2')
        net = ConvLayer(net, 16, [15, 15], name='conv2', pad='VALID')
        net = ConcatBlock(net, 112, 48, name='concat_block3')
        net = ConcatBlock(net, 64, 32, name='concat_block4')
        net = ConcatBlock(net, n_filters_mid, n_filters_mid, name='concat_block5')
        mid = net
        net = ConcatBlock(net, 32, 96, name='concat_block6')
        net = ConvLayer(net, 32, [17, 17], name='conv3', pad='VALID')
        net = ConvLayer(net, 64, [1, 1], name='conv4', pad='VALID')
        net = ConvLayer(net, n_filters_last, [1, 1], name='conv5', pad='VALID')
        last = net
        net_reg = ConvLayer(net, 1, [1, 1], name='out_reg', pad='VALID', is_last=True)
        net_sem = ConvLayer(net, 2, [1, 1], name='out_sem', pad='VALID', is_last=True)

    if is_return_feat:
        return {'reg': net_reg, 'sem': net_sem}, mid, last
    else:
        return {'reg': net_reg, 'sem': net_sem}

