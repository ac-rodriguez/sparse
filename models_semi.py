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

from flip_gradient import flip_gradient
def domain_discriminator(input, scope_name='domain_discriminator', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, return_feat = False):
    with tf.variable_scope(scope_name, reuse=reuse):

        x = flip_gradient(input)

        x = tf.layers.conv2d(x, 64, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 128, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 256, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        x1 = x
        x = tf.layers.conv2d(x, 256, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 2, kernel_size=4, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        if return_feat:
            return x, x1
        else:
            return x

def domain_discriminator_small(input, scope_name='domain_discriminator_single', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, return_feat = False):
    with tf.variable_scope(scope_name, reuse=reuse):

        x = flip_gradient(input)

        x = tf.layers.conv2d(x, 64, kernel_size=3, strides=2, padding='valid')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 32, kernel_size=3, strides=2, padding='valid')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        x = tf.layers.conv2d(x, 3, kernel_size=3, strides=2, padding='valid')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        x1 = x
        x = tf.layers.flatten(x)
        x = tf.layers.dense(x,2)
        # x = tf.layers.conv2d(x, 2, kernel_size=4, strides=1, padding='same')
        # x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        if return_feat:
            return x, x1
        else:
            return x

def decode(input, scope_name='decode', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, scale=8, n_feat_last = None):
    with tf.variable_scope(scope_name, reuse=reuse):
        n_feat = 128
        x = tf.layers.conv2d_transpose(input, 64, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        for i in range(int(np.log2(scale))):
            x = tf.layers.conv2d_transpose(x, 128, kernel_size=3, strides=2, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
            if n_feat_last is not None and i == int(np.log2(scale))-1:
                n_feat = n_feat_last
            x = tf.layers.conv2d_transpose(x, n_feat, kernel_size=3, strides=1, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x

def encode(input, scope_name='encode', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, scale=8):
    with tf.variable_scope(scope_name, reuse=reuse):

        x = tf.layers.conv2d(input, 64, kernel_size=3, strides=1, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

        for _ in range(int(np.log2(scale))):
            x = tf.layers.conv2d(x, 128, kernel_size=3, strides=2, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x

            x = tf.layers.conv2d(x, 128, kernel_size=3, strides=1, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x

def encode_same(input, scope_name='encode_same', is_training=True, is_bn=True, reuse=tf.AUTO_REUSE, is_small = True):
    with tf.variable_scope(scope_name, reuse=reuse):

        x = tf.layers.conv2d(input, 64, 3, activation=tf.nn.relu, padding='same')
        x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        iters = 1 if is_small else 3

        for _ in range(iters):
            x = tf.layers.conv2d(x, 128, 3, activation=tf.nn.relu, padding='same')
            x = bn_layer(x, activation_fn=tf.nn.leaky_relu, is_training=is_training) if is_bn else x
        return x


class SemisupModel(object):
    """Helper class for setting up semi-supervised training."""

    def __init__(self):
        """Initialize SemisupModel class.

        Creates an evaluation graph for the provided model_func.

        Args:
          model_func: Model function. It should receive a tensor of images as
              the first argument, along with the 'is_training' flag.
          num_labels: Number of taget classes.
          input_shape: List, containing input images shape in form
              [height, width, channel_num].
          test_in: None or a tensor holding test images. If None, a placeholder will
            be created.
        """

        # self.step = tf.train.get_or_create_global_step()
        # self.ema = tf.train.ExponentialMovingAverage(0.99, self.step)
        self.ema = tf.train.ExponentialMovingAverage(0.99)
        tf.moving_average_variables()
        self.loss = 0.0


    def add_semisup_loss(self, a, b, labels, walker_weight=1.0, visit_weight=1.0):
        """Add semi-supervised classification loss to the model.

        The loss constist of two terms: "walker" and "visit".

        Args:
          a: [N, emb_size] tensor with supervised embedding vectors.
          b: [M, emb_size] tensor with unsupervised embedding vectors.
          labels : [N] tensor with labels for supervised embeddings.
          walker_weight: Weight coefficient of the "walker" loss.
          visit_weight: Weight coefficient of the "visit" loss.
        """
        labels = tf.reshape(labels,[-1])
        a = tf.reshape(a,[-1, a.shape[-1]])
        b = tf.reshape(b,[-1, b.shape[-1]])
        equality_matrix = tf.equal(tf.reshape(labels, [-1, 1]), labels)
        equality_matrix = tf.cast(equality_matrix, tf.float32)
        p_target = (equality_matrix / tf.reduce_sum(
            equality_matrix, [1], keepdims=True))

        match_ab = tf.matmul(a, b, transpose_b=True, name='match_ab')
        p_ab = tf.nn.softmax(match_ab, name='p_ab')
        p_ba = tf.nn.softmax(tf.transpose(match_ab), name='p_ba')
        self.p_aba = tf.matmul(p_ab, p_ba, name='p_aba')

        self.create_walk_statistics(self.p_aba, equality_matrix)

        loss_aba = tf.losses.softmax_cross_entropy(
            p_target,
            tf.log(1e-8 + self.p_aba),
            weights=walker_weight,
            scope='loss_aba')
        tf.summary.scalar('loss/aba', loss_aba)
        self.loss += loss_aba
        visit_loss = self.get_visit_loss(p_ab, visit_weight)
        self.loss += visit_loss

        return self.loss

    def get_visit_loss(self, p, weight=1.0):
        """Add the "visit" loss to the model.

        Args:
          p: [N, M] tensor. Each row must be a valid probability distribution
              (i.e. sum to 1.0)
          weight: Loss weight.
        """
        visit_probability = tf.reduce_mean(
            p, [0], keepdims=True, name='visit_prob')
        t_nb = tf.shape(p)[1]
        visit_loss = tf.losses.softmax_cross_entropy(
            tf.fill([1, t_nb], 1.0 / tf.cast(t_nb, tf.float32)),
            tf.log(1e-8 + visit_probability),
            weights=weight,
            scope='loss_visit')
        tf.summary.scalar('loss/visit', visit_loss)
        return visit_loss


    def create_walk_statistics(self, p_aba, equality_matrix):
        """Adds "walker" loss statistics to the graph.

        Args:
          p_aba: [N, N] matrix, where element [i, j] corresponds to the
              probalility of the round-trip between supervised samples i and j.
              Sum of each row of 'p_aba' must be equal to one.
          equality_matrix: [N, N] boolean matrix, [i,j] is True, when samples
              i and j belong to the same class.
        """
        # Using the square root of the correct round trip probalilty as an estimate
        # of the current classifier accuracy.
        per_row_accuracy = 1.0 - tf.reduce_sum((equality_matrix * p_aba), 1) ** 0.5
        self.estimate_error = tf.reduce_mean(
            1.0 - per_row_accuracy, name=p_aba.name[:-2] + '_esterr')

        # self.add_vars()
        tf.summary.scalar('Stats_EstError', self.estimate_error)

    def add_average(self, variable):
        """Add moving average variable to the model."""
        apply_op = self.ema.apply([variable])
        return apply_op
        # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, apply_op)
        # average_variable = tf.identity(
        #     self.ema.average(variable), name=variable.name[:-2] + '_avg')
        # return average_variable
    #TODO implement ema variables for train/eval
    def add_vars(self):
        op1 = self.add_average(self.estimate_error)
        op2 = self.add_average(self.p_aba)
        # return tf.group((op1,op2))


    # def create_train_op(self, learning_rate):
    #     """Create and return training operation."""
    #
    #     slim.model_analyzer.analyze_vars(
    #         tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)
    #
    #     self.train_loss = tf.losses.get_total_loss()
    #     self.train_loss_average = self.add_average(self.train_loss)
    #
    #     tf.summary.scalar('Learning_Rate', learning_rate)
    #     tf.summary.scalar('Loss_Total_Avg', self.train_loss_average)
    #     tf.summary.scalar('Loss_Total', self.train_loss)
    #
    #     trainer = tf.train.AdamOptimizer(learning_rate)
    #
    #     self.train_op = slim.learning.create_train_op(self.train_loss, trainer)
    #     return self.train_op

    # def calc_embedding(self, images, endpoint):
    #     """Evaluate 'endpoint' tensor for all 'images' using batches."""
    #     batch_size = self.test_batch_size
    #     emb = []
    #     for i in xrange(0, len(images), batch_size):
    #         emb.append(endpoint.eval({self.test_in: images[i:i + batch_size]}))
    #     return np.concatenate(emb)

    # def classify(self, images):
    #     """Compute logit scores for provided images."""
    #     return self.calc_embedding(images, self.test_logit)
