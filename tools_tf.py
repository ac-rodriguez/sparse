import os, glob, shutil
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.contrib.tensorboard.plugins import projector
import tensorflow.contrib.slim as slim

import numpy as np
from tqdm import tqdm

import plots

import patches

def analyze_model():
    slim.model_analyzer.analyze_vars(
        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), print_info=True)

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
    if len(X.shape) == 3: X = tf.expand_dims(X, -1)
    return tf.nn.avg_pool(X, ksize=(1, scale, scale, 1),
                          strides=(1, scale, scale, 1), padding='VALID', name=name)


def median_pool(X, scale, name=None):
    if len(X.shape) == 3: X = tf.expand_dims(X, -1)

    patches = tf.extract_image_patches(X, [1, scale, scale, 1], [1, scale, scale, 1], 4 * [1], padding='VALID')
    median = tf.contrib.distributions.percentile(patches, 50, axis=-1)
    return tf.identity(tf.expand_dims(median, -1), name=name)


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


def inv_lr_decay(learning_rate, global_step, gamma, power, name=None):
    if global_step is None:
        raise ValueError("global_step is required for inv_decay.")
    with ops.name_scope(name, "InvDecay", \
                        [learning_rate, global_step, gamma, power]) as name:
        learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        gamma = math_ops.cast(gamma, dtype)
        power = math_ops.cast(power, dtype)

        def decayed_lr(global_step):
            global_step = math_ops.cast(global_step, dtype)
            # base = math_ops.multiply(gamma, global_step)
            # return math_ops.multiply( \
            #         learning_rate, math_ops.pow(1+base, -power), name=name)
            return tf.identity(learning_rate * ((1 + gamma * global_step) ** (-power)), name=name)

        if not context.executing_eagerly():
            decayed_lr = decayed_lr(global_step)
        return decayed_lr


class SessionHook(tf.train.SessionRunHook):

    def __init__(self, values, labels, num_batches=1):
        super(SessionHook, self).__init__()

        # self.iterator_initializer_func = None

        self._tensors = None

        # values = scopes['net_scope'] + '/' + scopes['emb_scope'] + '/' + scopes['emb_name'] + '/BiasAdd:0'
        # labels = scopes['metrics_scope'] + '/' + scopes['label_name'] + ':0'

        self._tensor_names = [values, labels]
        self._embeddings = [[], []]
        self.num_batches = num_batches
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
            self.sample_ind = np.random.choice(self.n_pixels, int(self.n_pixels * 0.1), replace=False)
        if self.iter <= self.num_batches:
            val_ = run_values[0][0].reshape(self.n_batch, -1, self.n_feat)[:, self.sample_ind, :].reshape(-1,
                                                                                                          self.n_feat)
            lab_ = run_values[0][1].reshape(self.n_batch, -1)[:, self.sample_ind].reshape(-1)
            # lab_ = np.mean(lab_,axis=0)
            self._embeddings[0].extend(val_)
            self._embeddings[1].extend(lab_)
            self.iter += 1

    def end(self, session):
        pass

    def get_embeddings(self):
        return {
            'values': self._embeddings[0],
            'labels': self._embeddings[1],
        }

    # def set_iterator_initializer(self, fun):
    #     self.iterator_initializer_func = fun


def get_embeddings(hook, Model_fn, suffix=''):
    embeddings = hook.get_embeddings()

    values = embeddings['values']
    labels = embeddings['labels']
    print('len embeddings', len(values))
    g_1 = tf.Graph()
    with g_1.as_default():

        embedding_var = tf.Variable(np.array(values), name='emb_values')

        path = os.path.join(Model_fn.model_dir, 'projector' + suffix)
        metadata = os.path.join(path, 'metadata.tsv')
        if not os.path.isdir(path):
            os.makedirs(path)

        with open(metadata, 'w+') as f:
            # f.write('Index\tLabel\n')
            for idx in range(len(labels)):
                f.write('{}\n'.format(labels[idx]))
            f.close()
        with tf.Session() as sess:
            sess.run(embedding_var.initializer)

            config = projector.ProjectorConfig()
            config.model_checkpoint_path = path + "/model_emb.ckpt"
            embedding = config.embeddings.add()
            embedding.tensor_name = embedding_var.name

            # Add metadata to the log
            embedding.metadata_path = metadata

            writer = tf.summary.FileWriter(path)
            projector.visualize_embeddings(writer, config)

            saver = tf.train.Saver([embedding_var])
            saver.save(sess, path + "/model_emb.ckpt")


def get_last_best_ckpt(path, folder='best_checkpoints'):
    list_of_files = glob.glob(path + '/{}/*.ckpt*index'.format(folder))
    assert len(list_of_files) > 0, 'no .ckpt found in {}/{}'.format(path, folder)
    last_file = max(list_of_files, key=os.path.getctime)
    return last_file.replace('.index', '')


def copy_last_ckpt(path, folder):
    destination_dir = os.path.join(path, folder)
    list_of_files = glob.glob(path + '/*.ckpt*index')
    assert len(list_of_files) > 0, 'no .ckpt found in {}'.format(path)

    last_file = max(list_of_files, key=os.path.getctime)
    list_to_copy = glob.glob(last_file.replace('.index', '') + '*')
    for file in list_to_copy:
        # print('copying {}'.format(file))
        shutil.copy(file, destination_dir)


class Checkpoint(object):
    dir = None
    file = None
    score = None
    path = None

    def __init__(self, path, score):
        self.dir = os.path.dirname(path)
        self.file = os.path.basename(path)
        self.score = score
        self.path = path


def batch_outerproduct(X, Y, n_feat=500, randomized=False, seeds=(100, 101)):
    X = tf.layers.flatten(X)
    Y = tf.layers.flatten(Y)

    if randomized:
        m1 = tf.random.normal(tf.TensorShape([X.shape[-1], n_feat]), seed=seeds[0])
        m2 = tf.random.normal(tf.TensorShape([Y.shape[-1], n_feat]), seed=seeds[1])

        X = tf.tensordot(X, m1, axes=1)
        Y = tf.tensordot(Y, m2, axes=1)
        denominator = (float(n_feat) ** 0.5)
    else:
        denominator = 1.0

    return tf.expand_dims(tf.expand_dims(X, 2) * tf.expand_dims(Y, 1) / denominator, -1)


def pair_distance(A, B, n_feat=500, randomized=False, seeds=(100, 101)):
    A = tf.layers.flatten(A)
    B = tf.layers.flatten(B)
    if randomized:
        m1 = tf.random.normal(tf.TensorShape([A.shape[-1], n_feat]), seed=seeds[0])
        m2 = tf.random.normal(tf.TensorShape([A.shape[-1], n_feat]), seed=seeds[1])

        A = tf.tensordot(A, m1, axes=1)
        B = tf.tensordot(B, m2, axes=1)
        denominator = (float(n_feat) ** 0.5)
    else:
        denominator = 1.0

    A2 = tf.reduce_sum(A * A, 1)
    A2 = tf.reshape(A2, [-1, 1, 1])

    B2 = tf.reduce_sum(B * B, 1)
    B2 = tf.reshape(B2, [-1, 1, 1])

    # outer prod
    AB = tf.expand_dims(A, 2) * tf.expand_dims(B, 1)

    D = tf.square(A2 - 2 * AB + B2)
    return D / denominator


def get_progress(args):
    progress = tf.cast(tf.train.get_global_step(), tf.float32) / (
            np.sum(args.train_patches) * args.epochs / float(args.batch_size))
    return progress


def evolving_lambda(args, height=1.0, lower=0.1, alpha=10.0):
    # height = 100.0
    # lower = 0.0
    # alpha = 10.0
    progress = get_progress(args)
    lambda_evol = 2.0 * height / (1.0 + tf.exp(-alpha * progress)) - height + lower
    return lambda_evol


def gaussian_kernel(size, mean, std, ):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)

    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))

    gauss_kernel = tf.einsum('i,j->ij',
                             vals,
                             vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


# def gaussian_noise_layer(input_layer, std):
#     noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
#     return input_layer + noise


def low_pass_filter(img, args, blur_probability=0.0, progressive=True):
    progress = get_progress(args)
    alpha = 5.0
    lower = 0.0001
    filter_size = 5
    height = float(args.scale)
    if progressive:

        scale_evol = 2.0 * height / (1.0 + tf.exp(-alpha * progress)) - height + lower

        sigma = scale_evol * tf.where(tf.greater(blur_probability, tf.random.uniform([1])), [1.0], [0.0])
    else:
        sigma = height
    # # Make Gaussian Kernel with desired specs.
    gauss_kernel = gaussian_kernel(5, (filter_size - 1) / 2.0, sigma)
    #
    gauss_kernel = gauss_kernel[:, :, tf.newaxis]
    kernel = tf.stack([gauss_kernel for _ in range(3)], 2)

    imout = tf.nn.depthwise_conv2d(img, kernel, strides=[1, 1, 1, 1], padding="SAME")
    return imout


def save_m(name, m):
    with open(name, 'w') as f:
        sorted_names = sorted(m.keys(), key=lambda x: x.lower())
        for key in sorted_names:
            value = m[key]
            f.write('%s:%s\n' % (key, value))


def predict_and_recompose(model, reader, input_fn, patch_generator, is_hr_pred, batch_size, type_,
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None):
    model_dir = model.model_dir
    save_dir = os.path.join(model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)

    # copy chkpt
    if chkpt_path is None and prefix != '':
        copy_last_ckpt(model_dir, prefix)

    f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, cmap='viridis', percentiles=(0,100))  #min=-1, max=2.0,
    nr_patches = patch_generator.nr_patches

    batch_idxs = int(np.ceil(nr_patches / batch_size))

    if is_hr_pred:
        patch = patch_generator.patch_h
        border = patch_generator.border_lab
        if 'val' in type_:
            ref_data = reader.val_h
        elif 'train' in type_:
            ref_data = reader.train_h
        else:
            ref_data = reader.test_h
    else:
        patch = patch_generator.patch_l
        border = patch_generator.border

        if 'val' in type_:
            ref_data = reader.val
        elif 'train' in type_:
            ref_data = reader.train
        else:
            ref_data = reader.test
    if is_reg:
        pred_r_rec = np.zeros(shape=([nr_patches, patch, patch, 1]))
    if is_sem:
        pred_c_rec = np.zeros(shape=([nr_patches, patch, patch]))
    if isinstance(ref_data,list):
        ref_data = ref_data[-1] # TODO fix choose data id
    ref_size = (ref_data.shape[1], ref_data.shape[0])

    # if args.domain is not None:
    #     hook = [tools.SessionHook(values="Embeddings:0", labels="EmbeddingsLabel:0", num_batches=10)]
    # else:
    hook = None
    preds_iter = model.predict(input_fn=input_fn, yield_single_examples=False, hooks=hook, checkpoint_path=chkpt_path)
    # checkpoint_path=tools.get_lastckpt(model_dir))  # ,predict_keys=['hr_hat_rgb'])

    print('Predicting {} Patches...'.format(nr_patches))
    for idx in tqdm(range(0, batch_idxs)):
        p_ = next(preds_iter)
        start = idx * batch_size
        # print(start,stop,stop-start)
        # print(p_['sem'].shape)
        if is_reg:
            stop = start + p_['reg'].shape[0]
            if stop > nr_patches:
                last_batch = nr_patches - start
                pred_r_rec[start:stop] = p_['reg'][0:last_batch]
            else:
                pred_r_rec[start:stop] = p_['reg']
        if is_sem:
            stop = start + p_['sem'].shape[0]
            if stop > nr_patches:
                last_batch = nr_patches - start
                pred_c_rec[start:stop] = np.argmax(p_['sem'][0:last_batch], axis=-1)
            else:
                pred_c_rec[start:stop] = np.argmax(p_['sem'], axis=-1)


    print(ref_size)
    ## Recompose RGB
    if is_reg:
        data_r_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border)
        if not return_array:
            np.save('{}/{}_reg_pred'.format(save_dir, type_), data_r_recomposed)
            plt_reg(data_r_recomposed, '{}/{}_reg_pred'.format(save_dir, type_))
    else:
        data_r_recomposed = None

    if is_sem:
        data_c_recomposed = patches.recompose_images(pred_c_rec, size=ref_size, border=border)
        if not return_array:
            np.save('{}/{}_sem_pred'.format(save_dir, type_), data_c_recomposed)
            plots.plot_labels(data_c_recomposed, '{}/{}_sem_pred'.format(save_dir, type_))
    else:
        data_c_recomposed = None

    if return_array:
        return data_r_recomposed, data_c_recomposed


class BestCheckpointCopier(tf.estimator.Exporter):
    checkpoints = None
    checkpoints_to_keep = None
    compare_fn = None
    name = None
    score_metric = None
    sort_key_fn = None
    sort_reverse = None

    def __init__(self, name='best_checkpoints', checkpoints_to_keep=5, score_metric='Loss/total_loss',
                 compare_fn=lambda x, y: x.score < y.score, sort_key_fn=lambda x: x.score, sort_reverse=False):
        self.checkpoints = []
        self.checkpoints_to_keep = checkpoints_to_keep
        self.compare_fn = compare_fn
        self.name = name
        self.score_metric = score_metric
        self.sort_key_fn = sort_key_fn
        self.sort_reverse = sort_reverse
        super(BestCheckpointCopier, self).__init__()

    def _copyCheckpoint(self, checkpoint):
        desination_dir = self._destinationDir(checkpoint)
        if not os.path.isdir(desination_dir): os.makedirs(desination_dir)

        for file in glob.glob(r'{}*'.format(checkpoint.path)):
            self._log('copying {} to {}'.format(file, desination_dir))
            shutil.copy(file, desination_dir)

    def _destinationDir(self, checkpoint):
        return os.path.join(checkpoint.dir, self.name)

    def _keepCheckpoint(self, checkpoint):
        self._log('keeping checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

        self.checkpoints.append(checkpoint)
        self.checkpoints = sorted(self.checkpoints, key=self.sort_key_fn, reverse=self.sort_reverse)

        self._copyCheckpoint(checkpoint)

    def _log(self, statement):
        tf.logging.info('[{}] {}'.format(self.__class__.__name__, statement))

    def _pruneCheckpoints(self, checkpoint):
        destination_dir = self._destinationDir(checkpoint)

        for checkpoint in self.checkpoints[self.checkpoints_to_keep:]:
            self._log('removing old checkpoint {} with score {}'.format(checkpoint.file, checkpoint.score))

            old_checkpoint_path = os.path.join(destination_dir, checkpoint.file)
            for file in glob.glob(r'{}*'.format(old_checkpoint_path)):
                self._log('removing old checkpoint file {}'.format(file))
                os.remove(file)

        self.checkpoints = self.checkpoints[0:self.checkpoints_to_keep]

    def _score(self, eval_result):
        return float(eval_result[self.score_metric])

    def _shouldKeep(self, checkpoint):
        return len(self.checkpoints) < self.checkpoints_to_keep or self.compare_fn(checkpoint, self.checkpoints[-1])

    def export(self, estimator, export_path, checkpoint_path, eval_result, is_the_final_export):
        self._log('export checkpoint {}'.format(checkpoint_path))

        score = self._score(eval_result)
        checkpoint = Checkpoint(path=checkpoint_path, score=score)

        if self._shouldKeep(checkpoint):
            self._keepCheckpoint(checkpoint)
            self._pruneCheckpoints(checkpoint)
        else:
            self._log('skipping checkpoint {}'.format(checkpoint.path))
