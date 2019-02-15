import os, glob, sys, re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize
from Queue import Queue
from threading import Thread, Lock
from functools import partial
import gdal

from plots import check_dims, plot_rgb

from read_geoTiff import readHR, readS2
import gdal_processing as gp


### ZRH 1 Sentinel-2 and Google-high res data

def interpPatches(image_20lr, ref_shape=None, squeeze=False, scale=None):
    image_20lr = check_dims(image_20lr)
    N, w, h, ch = image_20lr.shape

    if scale is not None:
        ref_shape = (int(w * scale), int(h * scale))
    image_20lr = np.rollaxis(image_20lr, -1, 1)

    data20_interp = np.zeros(((N, ch) + ref_shape)).astype(np.float32)
    for k in range(N):
        for w in range(ch):
            data20_interp[k, w] = resize(image_20lr[k, w] / 65000, ref_shape, mode='reflect') * 65000  # bicubic

    data20_interp = np.rollaxis(data20_interp, 1, 4)
    if squeeze:
        data20_interp = np.squeeze(data20_interp, axis=0)

    return data20_interp


def read_and_upsample_sen2(args, roi_lon_lat):
    data10, data20 = readS2(args, roi_lon_lat)
    data20 = interpPatches(data20, data10.shape[0:2], squeeze=True)

    data = np.concatenate((data10, data20), axis=2)

    size = np.prod(data.shape) * data.itemsize

    print('{:.2f} GB loaded in memory'.format(size / 1e9))
    return data


def read_labels(args, roi, roi1, is_HR=False):
    # if args.HR_file is not None:
    if is_HR:
        ds_file = args.HR_file
        scale = 1
    else:
        ds_file = os.path.join(os.path.dirname(args.LR_file), 'geotif', 'Band_B3.tif')
        scale = 10

    print(' [*] Reading Labels {}'.format(os.path.basename(args.points)))

    ds = gdal.Open(ds_file)
    print(' [*] Reading complete Area')

    lims_H = gp.to_xy_box(roi, ds, enlarge=2)
    print(' [*] Reading labeled Area')

    lims_H1 = gp.to_xy_box(roi1, ds, enlarge=2)
    # TODO check for predict if we have a smoothing of points
    labels = gp.rasterize_points_constrained(Input=args.points, refDataset=ds_file, lims=lims_H,
                                             lims1=lims_H1,
                                             scale=scale)  # DS is already at HR scale we do not need to upsample anything

    return np.expand_dims(labels, axis=2)


def read_and_upsample_test_file(path):
    # TODO finish implementation for zrh 1
    try:
        print('reading {} ...'.format(path))
        with np.load(path, mmap_mode='r') as data_:
            # points_new = data_['points10']
            data10 = data_['data10']
            data20 = data_['data20']
            cloud20 = data_['cld20']

    except IOError:

        print('{}.npz file could not be read/found'.format(path))
        return None

    if data20.shape[2] == 7:
        data20 = data20[..., 0:6]
        print('SCL removed from data20')

    print("# Bands 10m:{} 20m:{}".format(data10.shape[-1], data20.shape[-1]))

    if data10.shape != data20.shape:
        print("Performing Bicubic interpolation of 20m data... ")
        data20 = interpPatches(data20, ref_shape=data10.shape[0:2], squeeze=True)
        cloud20 = interpPatches(cloud20, ref_shape=data10.shape[0:2], squeeze=True)

    data10 = np.concatenate((data10, data20, cloud20), axis=2)
    del data20, cloud20

    N = data10.shape[0]
    print("N = {}".format(N))
    print('{:.2f} GB loaded in memory'.format(
        (np.prod(data10.shape) * data10.itemsize) / 1e9))

    return data10


def sub_set(data, select_bands):
    select_bands = ([x for x in re.split(',', select_bands)])

    all_bands = 'B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,CLD'
    all_bands = ([x for x in re.split(',', all_bands)])

    assert len(all_bands) == data.shape[-1], 'Not all bands were included in create_3A, subsetting won\'t work'

    band_id = [val in select_bands for val in all_bands]
    print('Subset of bands {} will be selected from {}'.format(select_bands, all_bands))

    return data[..., band_id]


class DataReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, args, is_training):
        '''Initialise a DataReader.

        Args:
          args: arguments from the train.py
        '''
        self.args = args
        self.is_training = is_training
        self.run_60 = False
        self.luminosity_scale = 9999.0
        self.batch = self.args.batch_size
        self.patch_l = self.args.patch_size
        self.patch_l_eval = self.args.patch_size_eval

        # TODO compute scale from the data
        self.scale = self.args.scale

        self.patch_h = self.args.patch_size * self.scale

        self.args.max_N = 1e5

        self.read_data(self.is_training)
        if self.is_training:

            if args.sigma_smooth:
                self.labels = map(lambda x: ndimage.gaussian_filter(x, sigma=args.sigma_smooth), self.labels)

            self.n_channels = self.train[0].shape[-1]

            self.patch_gen = PatchExtractor(dataset_low=self.train, dataset_high=self.train_h, label=self.labels,
                                            patch_l=self.patch_l, n_workers=4,max_queue_size=10, is_random=True, scale=args.scale)

            # featl,datah = self.patch_gen.get_inputs()
            # plt.imshow(datah[...,0:3])
            # im = plot_rgb(featl, file='', return_img=True)
            # im.show()
            #
            # im = plot_rgb(feath, file='', return_img=True)
            # im.show()


            print('Done')

            # self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
            #                                     patch_l=self.patch_l, n_workers=1, is_random=False, border=4,
            #                                     scale=args.scale)
            self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
                                                patch_l=self.patch_l_eval, n_workers=1, is_random=True, border=4,
                                                scale=args.scale)

            # featl,data_h = self.patch_gen_val.get_inputs()
            # plt.imshow(data_h[...,0:3])
            # im = plot_rgb(featl,file ='', return_img=True)
            # im.show()

            # value = self.input_fn(is_train=False)
            # # value = iter.get_next()
            # sess = tf.Session()
            # for i in range(5):
            #     val_ = sess.run(value)
            #     print(val_[0].shape, val_[1].shape)
            #
            # self.iter_val = self.patch_gen_val.get_iter()



        else:

            self.patch_gen_test = PatchExtractor(dataset_low=self.train, dataset_high=None, label=self.labels,
                                                 patch_l=self.patch_l, n_workers=1, is_random=False, border=4,
                                                 scale=args.scale)

            self.n_channels = self.train[0].shape[-1]

            # for _ in range(10):
            #     feat,_ = self.patch_gen_test.get_inputs()
            #
            #     # # plt.imshow(label.squeeze())
            #     im = plot_rgb(feat,file ='', return_img=True)
            #     im.show()
            #
            # print('done')

    def normalize(self, features, labels):

        features['feat_l'] -= self.mean_train
        features['feat_l'] /= self.std_train

        return features, labels

    def reorder_ds(self, data_l, data_h):
        if self.is_HR_labels:
            return {'feat_l': data_l,
                    'feat_h': data_h[..., 0:3]}, tf.expand_dims(data_h[..., -1], axis=-1)
        else:
            return {'feat_l': data_l[..., 0:self.n_channels],
                    'feat_h': data_h}, tf.expand_dims(data_l[..., -1], axis=-1)

    def read_data(self, is_training=True):

        if is_training:
            self.is_HR_labels = False
            print(' [*] Loading Train data ')
            self.train_h = readHR(self.args,
                                  roi_lon_lat=self.args.roi_lon_lat_tr) if self.args.HR_file is not None else None
            self.train = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)
            self.labels = read_labels(self.args, roi=self.args.roi_lon_lat_tr, roi1=self.args.roi_lon_lat_tr_lb,
                                      is_HR=self.is_HR_labels)

            print(' [*] Loading Validation data ')
            self.val_h = readHR(self.args,
                                roi_lon_lat=self.args.roi_lon_lat_val) if self.args.HR_file is not None else None
            self.val = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_val)
            self.labels_val = read_labels(self.args, roi=self.args.roi_lon_lat_val, roi1=self.args.roi_lon_lat_val_lb,
                                          is_HR=self.is_HR_labels)

            # sum_train = np.expand_dims(self.train.sum(axis=(0, 1)),axis=0)
            # self.N_pixels = self.train.shape[0] * self.train.shape[1]

            self.mean_train = self.train.mean(axis=(0, 1))
            self.std_train = self.train.std(axis=(0, 1))

            print(str(self.mean_train))
            self.nb_bands = self.train.shape[-1]

            # TODO check why there is a difference in the shapes
            scale = self.args.scale
            if self.train_h is not None:
                x_shapes, y_shapes = self.compute_shapes(dset_h=self.train_h, dset_l=self.train, scale=scale)

                print(
                ' Train shapes: \n\tBefore:\tLow:({}x{})\tHigh:({}x{})'.format(self.train.shape[0], self.train.shape[1],
                                                                               self.train_h.shape[0],
                                                                               self.train_h.shape[1]))
                # Reduce data to the enlarged 10m pixels
                self.train_h = self.train_h[0:int(scale * x_shapes), 0:int(scale * y_shapes), :]
                self.train = self.train[0:int(x_shapes), 0:int(y_shapes), :]
                if self.is_HR_labels:
                    self.labels = self.labels[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                print('\tAfter:\tLow:({}x{})\tHigh:({}x{})'.format(self.train.shape[0], self.train.shape[1],
                                                                   self.train_h.shape[0], self.train_h.shape[1]))

                x_shapes, y_shapes = self.compute_shapes(dset_h=self.labels_val, dset_l=self.val, scale=scale)

                print(' Val shapes: \n\tBefore:\tLow:({}x{})\tHigh:({}x{})'.format(self.val.shape[0], self.val.shape[1],
                                                                                   self.val_h.shape[0],
                                                                                   self.val_h.shape[1]))

                # Reduce data to the enlarged 10m pixels
                self.val_h = self.val_h[0:int(scale * x_shapes), 0:int(scale * y_shapes), :]
                self.val = self.val[0:int(x_shapes), 0:int(y_shapes), :]
                if self.is_HR_labels:
                    self.labels_val = self.labels_val[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                print('\tAfter:\tLow:({}x{})\tHigh:({}x{})'.format(self.val.shape[0], self.val.shape[1],
                                                                   self.val_h.shape[0], self.val_h.shape[1]))


        else:

            print(' [*] Loading data for Prediction ')
            # self.train_h = readHR(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)
            self.train = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)
            if self.args.points is not None:
                self.labels = read_labels(self.args, roi=self.args.roi_lon_lat_tr, roi1=self.args.roi_lon_lat_tr_lb)
            else:
                self.labels = None

            self.nb_bands = self.train.shape[-1]

            # TODO check why there is a difference in the shapes
            scale = self.args.scale

            print(
                ' Predict shapes: \n\tBefore:\tLow:({}x{})'.format(self.train.shape[0], self.train.shape[1]))
            # Reduce data to the enlarged 10m pixels
            if self.labels is not None:
                x_shapes, y_shapes = self.compute_shapes(dset_h=self.labels, dset_l=self.train, scale=scale)

                self.train = self.train[0:int(x_shapes), 0:int(y_shapes), :]
                self.labels = self.labels[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                print('\tAfter:\tLow:({}x{})\tHigh:({}x{})'.format(self.train.shape[0], self.train.shape[1],
                                                                   self.labels.shape[0], self.labels.shape[1]))

    def compute_shapes(self, scale, dset_h, dset_l):

        enlarge = lambda x: int(x / scale) * scale
        x_h, y_h, _ = map(enlarge, dset_h.shape)
        x_l, y_l, _ = dset_l.shape
        x_shape, y_shape = min(int(x_h / scale), x_l), min(int(y_h / scale), y_l)
        return x_shape, y_shape

    def input_fn(self, is_train=True):
        # np.random.seed(99)

        tf.Variable(self.mean_train.astype(np.float32), name='mean_train', trainable=False, validate_shape=True,
                    expected_shape=tf.shape([self.n_channels]))
        # tf.Variable(self.luminosity_scale, name='scale_preprocessing', trainable=False, expected_shape=tf.shape(1))
        tf.Variable(self.std_train.astype(np.float32), name='std_train', trainable=False,
                    expected_shape=tf.shape([self.n_channels]))

        tf.constant(self.mean_train.astype(np.float32), name='mean_train_k')
        # tf.constant(self.luminosity_scale, name='scale_preprocessing_k')
        tf.constant(self.std_train.astype(np.float32), name='std_train_k')
        if is_train:
            gen_func = self.patch_gen.get_iter
            patch_l, patch_h = self.patch_l, self.patch_h
        else:
            gen_func = self.patch_gen_val.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)

        if self.is_HR_labels:
            n_low = self.n_channels
            n_high = 4
        else:
            n_low = self.n_channels + 1
            n_high = 3
        ds = tf.data.Dataset.from_generator(
            gen_func, (tf.float32, tf.float32),
            (
                tf.TensorShape([patch_l, patch_l,
                                n_low]),
                tf.TensorShape([patch_h, patch_h,
                                n_high])
            ))
        ds = ds.batch(self.batch).map(self.reorder_ds, num_parallel_calls=6).map(self.normalize, num_parallel_calls=6)

        ds = ds.prefetch(buffer_size=self.args.batch_size*2)

        return ds

    def get_input_fn(self):

        input_fn = partial(self.input_fn, is_train=True)
        input_fn_val = partial(self.input_fn, is_train=False)

        return input_fn, input_fn_val

    def input_fn_test(self):

        self.mean_train = tf.Variable(np.zeros(self.train.shape[-1]), name='mean_train', trainable=False,
                                      dtype=tf.float32)
        self.luminosity_scale = tf.Variable(9999.0, name='scale_preprocessing', trainable=False, dtype=tf.float32)
        gen_func = self.patch_gen_test.get_iter

        if self.labels is not None:
            ds = tf.data.Dataset.from_generator(
                gen_func, (tf.float32, tf.float32),
                (
                    tf.TensorShape([self.patch_l, self.patch_l,
                                    self.n_channels]),
                    tf.TensorShape([self.patch_h, self.patch_h,
                                    1])
                ))
            ds = ds.map(self.reorder_ds).map(self.normalize)
        else:
            ds = tf.data.Dataset.from_generator(
                gen_func, (tf.float32, tf.float32),
                (
                    tf.TensorShape([self.patch_l, self.patch_l,
                                    self.n_channels]),
                    tf.TensorShape([None])
                ))
            ds = ds.map(self.normalize)

        ds = ds.batch(self.batch).prefetch(buffer_size=10)

        return ds


class PatchExtractor:
    def __init__(self, dataset_low, dataset_high, label, patch_l=16, max_queue_size=4, n_workers=1, is_random=True,
                 border=None, scale=None, return_corner=False, keep_edges=True):

        self.d_l = dataset_low
        self.d_h = dataset_high
        self.label = label
        if self.d_l.shape[0:2] == self.label.shape[0:2]:
            self.is_HR_label = True
        else:
            self.is_HR_label = False

        self.is_random = is_random
        self.border = border
        self.scale = scale
        self.nr_patches = 1e5

        self.return_corner = return_corner
        self.keep_edges = keep_edges

        self.patch_l = patch_l
        assert self.patch_l <= self.d_l.shape[0] and self.patch_l <= self.d_l.shape[1], \
            ' patch of size {} is bigger than ds_l {}'.format(self.patch_l, self.d_l.shape)

        self.patch_h = patch_l * scale
        if self.border is not None:
            assert self.patch_l > self.border * 2

        if not self.is_random:
            self.compute_tile_ranges()

        else:
            n_x, self.n_y = np.subtract(self.d_l.shape[0:2], self.patch_l)
            # Corner is always computed in low_res data
            max_patches = min(self.nr_patches, n_x*self.n_y)
            # Corner is always computed in low_res data
            self.indices = np.random.choice(n_x * self.n_y, size=int(max_patches), replace=False)
            self.rand_ind = 0
        self.lock = Lock()
        self.inputs_queue = Queue(maxsize=max_queue_size)
        self._start_batch_makers(n_workers)

    def _start_batch_makers(self, number_of_workers):
        for w in range(number_of_workers):
            worker = Thread(target=self._inputs_producer)
            worker.setDaemon(True)
            worker.start()

    def _inputs_producer(self):
        if self.is_random:
            while True:
                self.inputs_queue.put(self.get_random_patches())
        else:

            while True:
                i = 0
                with self.lock:
                    # for data_id in range(len(self.d_l)):
                    for ii in self.range_i:
                        for jj in self.range_j:
                            # print(i, ii, jj)
                            self.inputs_queue.put(self.get_patches(xy_corner=(ii, jj)))  # + (i,)
                            i += 1
                    print 'starting over Val set {}'.format(i)

    def get_patches(self, xy_corner):

        if self.scale is None:
            scale = 1
        else:
            scale = self.scale
        x_l, y_l = map(int, xy_corner)
        x_h, y_h = x_l * scale, y_l * scale

        if self.d_h is not None:
            patch_h = self.d_h[x_h:x_h + self.patch_h,
                      y_h:y_h + self.patch_h]

            assert patch_h.shape == (self.patch_h, self.patch_h, self.d_h.shape[-1],), \
                'Shapes: Dataset={} Patch={} xy_corner={}'.format(self.d_h.shape, patch_h.shape, (x_h, y_h))
        else:
            patch_h = None
        if self.label is not None:
            if self.is_HR_label:
                label = self.label[x_h:x_h + self.patch_h,
                        y_h:y_h + self.patch_h]

                assert label.shape == (self.patch_h, self.patch_h, self.label.shape[-1],), \
                    'Shapes: Dataset={} Patch={} xy_corner={}'.format(self.label.shape, label.shape, (x_h, y_h))
            else:
                label = self.label[x_l:x_l + self.patch_l,
                        y_l:y_l + self.patch_l]

                assert label.shape == (self.patch_l, self.patch_l, self.label.shape[-1],), \
                    'Shapes: Dataset={} Patch={} xy_corner={}'.format(self.label.shape, label.shape, (x_h, y_h))

        else:
            label = None
        if self.scale is not None:

            patch_l = self.d_l[x_l:x_l + self.patch_l,
                      y_l:y_l + self.patch_l]
        else:
            patch_l = None

        if self.is_HR_label:
            data_h = np.dstack((patch_h, label)) if patch_h is not None else label
        else:
            data_h = patch_h
            patch_l = np.dstack((patch_l, label))
        if self.return_corner:
            return patch_l, data_h, xy_corner
        else:

            return patch_l, data_h

    def get_random_patches(self):


        with self.lock:
            ind = self.indices[np.mod(self.rand_ind,len(self.indices))]
            # print ' rand_index={}'.format(self.rand_ind)
            self.rand_ind+=1
        # ind = 50
        corner_ = divmod(ind, self.n_y)

        return self.get_patches(corner_)

    def compute_tile_ranges(self):

        borders = (self.border, self.border)
        borders_h = (self.border * self.scale, self.border * self.scale)

        self.range_i = [None] * len(self.d_l)
        self.range_j = [None] * len(self.d_l)
        self.nr_patches = [None] * len(self.d_l)

        # for i, data_ in enumerate(self.d_l):

        data_ = np.pad(self.d_l, (borders, borders, (0, 0)), mode='symmetric')

        range_i = np.arange(0, (data_.shape[0] - 2 * self.border) // (self.patch_l - 2 * self.border)) * (
            self.patch_l - 2 * self.border)
        range_j = np.arange(0, (data_.shape[1] - 2 * self.border) // (self.patch_l - 2 * self.border)) * (
            self.patch_l - 2 * self.border)

        x_excess = np.mod(data_.shape[0] - 2 * self.border, self.patch_l - 2 * self.border)
        y_excess = np.mod(data_.shape[1] - 2 * self.border, self.patch_l - 2 * self.border)

        if not (x_excess == 0):
            if self.keep_edges:
                range_i = np.append(range_i, (data_.shape[0] - self.patch_l))
            else:
                print('{} pixels in x axis will be discarded'.format(x_excess))
        if not (y_excess == 0):
            if self.keep_edges:
                range_j = np.append(range_j, (data_.shape[1] - self.patch_l))
            else:
                print('{} pixels in y axis will be discarded'.format(y_excess))

        # nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)
        nr_patches = len(range_i) * len(range_j)

        print('   Shapes Original = {}'.format(data_.shape))
        print('   Shapes Patched (low-res) Dataset = {}'.format(
            [nr_patches, self.patch_l, self.patch_l, data_.shape[-1]]))

        self.d_l = data_

        self.nr_patches = nr_patches
        self.range_i = range_i
        self.range_j = range_j

        self.d_h = np.pad(self.d_h, (borders_h, borders_h, (0, 0)), mode='symmetric') if self.d_h is not None else None

        if self.is_HR_label:
            self.label = np.pad(self.label, (borders_h, borders_h, (0, 0)),
                                mode='symmetric') if self.label is not None else None
        else:
            self.label = np.pad(self.label, (borders, borders, (0, 0)),
                                mode='symmetric') if self.label is not None else None

    def get_inputs(self):
        return self.inputs_queue.get()

    def get_iter(self):
        while True:
            yield self.get_inputs()[0:2]
