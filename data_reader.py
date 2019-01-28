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
# from data_utils import observe_image
from read_geoTiff import readHR, readS2
import gdal_processing as gp
### ZRH 1 Sentinel-2 and Google-high res data

def interpPatches(image_20lr, ref_shape=None, squeeze = False,scale = None):

    image_20lr = check_dims(image_20lr)
    N, w,h, ch = image_20lr.shape

    if scale is not None:
        ref_shape = (int(w*scale),int(h*scale))
    image_20lr = np.rollaxis(image_20lr,-1,1)

    data20_interp = np.zeros(((N, ch) + ref_shape)).astype(np.float32)
    for k in range(N):
        for w in range(ch):
            data20_interp[k, w] = resize(image_20lr[k, w] / 65000, ref_shape, mode='reflect') * 65000  # bicubic

    data20_interp = np.rollaxis(data20_interp,1,4)
    if squeeze:
        data20_interp = np.squeeze(data20_interp, axis= 0)

    return data20_interp


def read_and_upsample_sen2(args, roi_lon_lat):

    data10, data20 = readS2(args, roi_lon_lat)
    data20 = interpPatches(data20, data10.shape[0:2], squeeze=True)

    data = np.concatenate((data10, data20), axis=2)

    size = np.prod(data.shape) * data.itemsize

    print('{:.2f} GB loaded in memory'.format(size / 1e9))
    return data

def read_labels(args, roi, roi1):
    dsH = gdal.Open(args.HR_file)
    lims_H = gp.to_xy_box(roi, dsH, enlarge=2)
    lims_H1 = gp.to_xy_box(roi1, dsH, enlarge=2)

    labels = gp.rasterize_points_constrained(Input=args.points, refDataset=args.HR_file, lims=lims_H,
                                                  lims1=lims_H1, scale=2)

    return labels


def read_hig_res(filename):

    data = np.load(filename)
    size = np.prod(data.shape) * data.itemsize
    print(' Shapes Data={}'.format(data.shape))

    print('{:.2f} GB loaded in memory'.format(np.sum(size) / 1e9))
    return data


def read_and_upsample_test_file(path):
    # TODO finish implementation for zrh 1
    try:
        print('reading {} ...'.format(path))
        with np.load(path,mmap_mode='r') as data_:
            # points_new = data_['points10']
            data10 = data_['data10']
            data20 = data_['data20']
            cloud20 = data_['cld20']

    except IOError:

        print('{}.npz file could not be read/found'.format(path))
        return None


    if data20.shape[2] == 7:
        data20 = data20[...,0:6]
        print('SCL removed from data20')

    print("# Bands 10m:{} 20m:{}".format(data10.shape[-1], data20.shape[-1]))

    if data10.shape != data20.shape:
        print("Performing Bicubic interpolation of 20m data... ")
        data20 = interpPatches(data20, ref_shape=data10.shape[0:2], squeeze=True)
        cloud20 = interpPatches(cloud20, ref_shape=data10.shape[0:2], squeeze=True)

    data10 = np.concatenate((data10, data20, cloud20), axis = 2)
    del data20, cloud20

    N = data10.shape[0]
    print("N = {}".format(N))
    print('{:.2f} GB loaded in memory'.format(
        (np.prod(data10.shape) * data10.itemsize) / 1e9))

    return data10

def sub_set(data, select_bands):

    select_bands = ([x for x in re.split(',',select_bands) ])

    all_bands = 'B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,CLD'
    all_bands = ([x for x in re.split(',', all_bands)])

    assert len(all_bands) == data.shape[-1], 'Not all bands were included in create_3A, subsetting won\'t work'

    band_id = [val in select_bands for val in all_bands]
    print('Subset of bands {} will be selected from {}'.format(select_bands,all_bands))

    return data[...,band_id]



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
        self.luminosity_scale = 2000 #9999.0
        self.batch = self.args.batch_size
        self.patch_l = self.args.patch_size

        # TODO compute it from the data
        self.scale = self.args.scale

        self.patch_h = self.args.patch_size * self.scale

        self.args.max_N = 1e5

        if self.is_training:

            # self.path = os.path.join(self.args.data_dir, 'train')  #  + str(self.args.patch_size))
            # self.path_val = self.path.replace('train','validation')
            #
            # if args.scale_points > 1:
            #     self.path = self.path + '_scale{:.1f}'.format(self.args.scale_points)
            #     self.path_val = self.path_val + '_scale{:.1f}'.format(self.args.scale_points)

            self.read_data()


            if args.sigma_smooth:
                self.labels = map(lambda x: ndimage.gaussian_filter(x, sigma=args.sigma_smooth),self.labels)

            weights = self.N_pixels / np.float32(np.sum(self.N_pixels))

            self.n_channels = self.train[0].shape[-1]



            self.patch_gen = PatchExtractor1(dataset_low=self.train, dataset_high=self.train_h,
                                             batch_size=self.batch, patch_l=self.patch_l, weights=weights,
                                             is_random=True, scale=args.scale)

            self.batch_val = 1
            self.patch_gen_val = PatchExtractor1(dataset_low=self.val_tr, dataset_high=self.val_h,
                                                 batch_size=self.batch_val, patch_l=self.patch_l, n_workers=1,
                                                 is_random=False, border=4, scale=args.scale)


            print('Done')

            # feat, label = self.patch_gen_val.get_inputs()
            # plt.imshow(label.squeeze())
            # im = plot_rgb(feat,file ='', return_img=True)
            # im.show()
            #
            #
            # feat1, label1 = self.patch_gen.get_inputs()
            # plt.imshow(label1[0].squeeze())
            # im = plot_rgb(feat1[0],file ='', return_img=True)
            # im.show()

            # value = self.iter_val.get_next()
            # sess = tf.Session()
            # for i in range(5):
            #     val_ = sess.run(value)
            #     print(val_[0].shape, val_[1].shape)

        else:
            self.path = self.args.data_dir

            self.read_data(is_training=False)
            self.n_channels = self.train[0].shape[-1]

            batch_val = 1
            border = 4
            self.patch_gen_test = PatchExtractor1(dataset_low=self.train, batch_size=batch_val, patch_l=self.patch_l,
                                                 max_queue_size=batch_val * self.batch * 2, is_random=False,
                                                 border=4, scale=args.scale)

            # for _ in range(10):
            #     feat,_ = self.patch_gen_test.get_inputs()
            #
            #     # # plt.imshow(label.squeeze())
            #     im = plot_rgb(feat,file ='', return_img=True)
            #     im.show()
                #
            # print('done')


    def preprocess(self,features, label):

        features = tf.concat((features[:, 0, ...], features[:, 1, ...], features[:, 2, ...], features[:, 3, ...]),
                             3)
        return features, label
    def normalize(self, img, label):

        # img, label = random_patches_image_and_labels(img,label,self.input_size,self.input_size_w, nb_bands=self.train.shape[-1])
        img /= self.luminosity_scale
        # img -= self.mean_train
        # # img /= tf.san
        label /= self.luminosity_scale
        # label -= self.mean_train
        # img = img / (self.SCALE + img)
        # label = label / (self.SCALE + label)

        return img, label
    def normalize1(self, feat, label):

        # img, label = random_patches_image_and_labels(img,label,self.input_size,self.input_size_w, nb_bands=self.train.shape[-1])
        feat /= self.luminosity_scale

        # Remove for now all the other channels from features
        # feat = feat[..., 0:3]

        # img -= self.mean_train
        # # img /= tf.san
        label = tf.cast(label,tf.float32) / 255.0
        # label -= self.mean_train
        # img = img / (self.SCALE + img)
        # label = label / (self.SCALE + label)

        return feat, label
    def read_data(self, is_training = True):

        if is_training:

            print(' [*] Loading Train data ')
            self.train_h = readHR(self.args,roi_lon_lat=self.args.roi_lon_lat_tr)
            self.train = read_and_upsample_sen2(self.args,roi_lon_lat=self.args.roi_lon_lat_tr)
            self.labels = read_labels(self.args, roi=self.args.roi_lon_lat_tr,roi1=self.args.roi_lon_lat_tr_lb)

            print(' [*] Loading Validation data ')
            self.val_h = readHR(self.args, roi_lon_lat=self.args.roi_lon_lat_val)
            self.val = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_val)
            self.labels_val = read_labels(self.args, roi=self.args.roi_lon_lat_val, roi1=self.args.roi_lon_lat_val_lb)


            sum_train = self.train.sum(axis=(0, 1))
            self.N_pixels = self.train.shape[0] * self.train.shape[1]

            self.mean_train = np.sum(sum_train, axis=0) / (np.sum(self.N_pixels) * self.luminosity_scale)
            # self.std = np.sum(sum_train, axis=0) / (np.sum(self.N_pixels) * self.SCALE)
            self.nb_bands = self.mean_train.shape[0]

            # if self.args.data == 'zrh1':
            # TODO check why there is a difference in the shapes
            scale = self.args.scale

            x_shapes, y_shapes = self.compute_shapes(dset_h=self.train_h,dset_l=self.train,scale=scale)
            # self.train = map(lambda x: interpPatches(x, scale=scale, squeeze=True),self.train)

            # Reduce data to the enlarged 10m pixels
            self.train_h = self.train_h[0:int(scale*x_shapes),0:int(scale*y_shapes),:]
            self.train = self.train[0:int(x_shapes), 0:int(y_shapes), :]
            self.labels = self.labels[0:int(scale * x_shapes), 0:int(scale * y_shapes)]


            x_shapes, y_shapes = self.compute_shapes(dset_h=self.val_h,dset_l=self.val,scale=scale)
            # self.val_tr = map(lambda x: interpPatches(x, scale=scale, squeeze=True), self.val_tr)

            # Reduce data to the enlarged 10m pixels
            self.val_h = self.val_h[0:int(scale*x_shapes),0:int(scale*y_shapes),:]
            self.val = self.val[0:int(x_shapes),0:int(y_shapes),:]
            self.labels_val = self.labels_val[0:int(scale * x_shapes), 0:int(scale * y_shapes)]


            # print(' upsampled lowres data...')

        else:

            # TODO
            sys.exit(1)
            if self.args.data == 'zrh':

                self.train = read_and_upsample_test_file(self.path)
            elif self.args.data == 'zrh1':
                roi_filter_val = '8.405-47.34-8.631-47.3137'

                self.train = read_and_upsample_sen2(self.path, roifilter=roi_filter_val)



    def compute_shapes(self,scale,dset_h,dset_l):

        enlarge = lambda x: int(x / scale) * scale
        # enlarge_shapes = lambda x: (enlarge(x.shape[0]), enlarge(x.shape[1]))
        get_shapes = lambda x: (x.shape[0], x.shape[1])

        x_shapes, y_shapes = zip(*map(get_shapes, dset_l))
        x_shapes = int(np.min(x_shapes + (enlarge(dset_h.shape[0]) / scale,)))
        y_shapes = int(np.min(y_shapes + (enlarge(dset_h.shape[1] / scale),)))
        return x_shapes, y_shapes


    def input_fn_zrh(self, is_train=True):
        # np.random.seed(99)

        tf.Variable(self.mean_train, name='mean_train', trainable=False)
        tf.Variable(self.luminosity_scale, name='scale_preprocessing', trainable=False)

        if is_train:

            ds = tf.data.Dataset.from_generator(
                self.patch_gen.get_iter, (tf.float32, tf.float32),
                (
                    tf.TensorShape([None, 4, self.patch_l, self.patch_l,
                                    self.n_channels]),
                    tf.TensorShape([None, self.patch_h, self.patch_h, self.n_channels])))

            ds = ds.map(self.normalize).map(self.preprocess)

            iter = ds.make_one_shot_iterator()

            return iter.get_next()
        else:

            ds_val = tf.data.Dataset.from_generator(
                self.patch_gen_val.get_iter, (tf.float32, tf.float32),
                (tf.TensorShape(
                    [None, 4, self.patch_l , self.patch_l , self.n_channels]),
                 tf.TensorShape([None, self.patch_h, self.patch_h, self.n_channels])))

            ds_val = ds_val.map(self.normalize).map(self.preprocess)
            ds_val = ds_val.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)))
            ds_val = ds_val.batch(self.batch)

            iter_val = ds_val.make_one_shot_iterator()

            return iter_val.get_next()

    def input_fn_zrh1(self, is_train=True):
        # np.random.seed(99)

        tf.Variable(self.mean_train, name='mean_train', trainable=False)
        tf.Variable(self.luminosity_scale, name='scale_preprocessing', trainable=False)

        if is_train:

            ds = tf.data.Dataset.from_generator(
                self.patch_gen.get_iter, (tf.float32, tf.uint8),
                (   tf.TensorShape([None, self.patch_l, self.patch_l, self.n_channels]),
                    tf.TensorShape([None, self.patch_h, self.patch_h, 3])))

            ds = ds.map(self.normalize1)

            iter = ds.make_one_shot_iterator()

            return iter.get_next()
        else:

            ds_val = tf.data.Dataset.from_generator(
                self.patch_gen_val.get_iter, (tf.float32, tf.uint8),
                (tf.TensorShape([self.batch_val, self.patch_l, self.patch_l, self.n_channels]),
                 tf.TensorShape([self.batch_val, self.patch_h, self.patch_h, 3])))

            ds_val = ds_val.map(self.normalize1)
            ds_val = ds_val.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)))
            ds_val = ds_val.batch(self.batch)

            iter_val = ds_val.make_one_shot_iterator()

            return iter_val.get_next()


    def get_input_fn(self):
        if self.args.data == 'zrh':
            input_fn = partial(self.input_fn_zrh, is_train=True)
            input_fn_val = partial(self.input_fn_zrh, is_train=False)
        elif self.args.data == 'zrh1':
            input_fn = partial(self.input_fn_zrh1, is_train=True)
            input_fn_val = partial(self.input_fn_zrh1, is_train=False)
        else:
            print('data not defined')
            return None, None

        return input_fn,input_fn_val
    def input_fn_test(self):

        tf.Variable(np.zeros(self.train[0].shape[-1]), name='mean_train', trainable=False)
        tf.Variable(9999.0, name='scale_preprocessing', trainable=False)

        # changing generator to yield only the patch
        def gen():
            # for _ in range(0,self.patch_gen_test.nr_patches[0]):
            while True:
                a = self.patch_gen_test.get_inputs()
                # print 'got element {}'.format(a[2])
                yield a[0]

        self.batch_val = 1

        self.ds_test = tf.data.Dataset.from_generator(
            gen, tf.float32,
            (tf.TensorShape([self.batch_val, self.patch_l, self.patch_l, self.n_channels])))


        self.ds_test = self.ds_test.map(lambda x: tf.squeeze(x / self.luminosity_scale, axis=0))
        self.ds_test = self.ds_test.batch(self.batch)

        self.iter_test = self.ds_test.make_one_shot_iterator()
        return self.iter_test.get_next()

class PatchExtractor:
    def __init__(self, dataset, batch_size=100, patch_l=16, max_queue_size=4, n_workers=8, weights=None, is_random=True,
                 border=None, scale=None, return_corner=False, keep_edges=True):

        self.dataset = dataset
        self.is_random = is_random
        self.border = border
        self.scale = scale
        self.batch_size = batch_size
        self.weights = weights
        self.return_corner = return_corner
        self.keep_edges = keep_edges

        self.patch_l = patch_l
        if scale is not None:
            self.patch_h = patch_l * scale

        if not isinstance(self.dataset,list):
            self.dataset = [self.dataset]

        if not self.is_random:
            self.compute_tile_ranges()
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
                self.inputs_queue.put(self.get_random_batch())
        else:

            while True:
                i = 0
                with self.lock:
                    for data_id in range(len(self.dataset)):
                        for ii in self.range_i[data_id]:
                            for jj in self.range_j[data_id]:
                                # print(i, ii, jj)
                                self.inputs_queue.put(self.get_patch_and_observe(xy_corner=(ii, jj), id=data_id)) # + (i,)
                                i += 1
                    print 'starting over Val set {}'.format(i)


    def get_patch_and_observe(self, xy_corner, id=0):

        x, y = xy_corner

        label_patch = self.dataset[id][x:x+self.patch_h,
                             y:y+self.patch_h]
        assert label_patch.shape == (self.patch_h, self.patch_h, self.dataset[id].shape[-1]),\
            'Shapes: Dataset={} Patch={} xy_corner={}'.format(self.dataset[id].shape,label_patch.shape, xy_corner)

        if self.scale is not None:
            lr = []
            for j, coord_ in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                temp = observe_image(label_patch, coord_, scale=self.scale)
                lr.append(temp)
            features = np.stack(lr)
            if not self.is_random:
                label_patch = np.expand_dims(label_patch,axis=0)
                features = np.expand_dims(features,axis=0)
        else:
            features = None
        if self.return_corner:
            return features, label_patch, xy_corner
        else:
            return features, label_patch

    def get_random_batch(self):
        feature_sets = []
        label_sets = []

        if len(self.dataset) > 1:
            # We choose one of the datasets weighted on the relative number of pixels they have
            id = np.random.choice(len(self.dataset),p=self.weights)
        else:
            id = 0

        n_x, n_y = np.subtract(self.dataset[id].shape[0:2], self.patch_h)

        ind = np.random.choice(n_x * n_y, int(self.batch_size), replace=False)

        for i in range(self.batch_size):
            corner_ = divmod(ind[i], n_y)
            feat_patch, label_patch = self.get_patch_and_observe(corner_, id)
            feature_sets.append(feat_patch)
            label_sets.append(label_patch)

        return (np.array(feature_sets), np.array(label_sets))

    def compute_tile_ranges(self):

        borders = (self.border, self.border)

        self.range_i = [None] * len(self.dataset)
        self.range_j = [None] * len(self.dataset)
        self.nr_patches = [None] * len(self.dataset)

        for i, data_ in enumerate(self.dataset):

            data_ = np.pad(data_, (borders, borders, (0, 0)), mode='symmetric')

            # patchesAlongi = (data_.shape[0] - 2 * self.border) // (self.patch_size[0] - 2 * self.border)
            # patchesAlongj = (data_.shape[1] - 2 * self.border) // (self.patch_size[1] - 2 * self.border)


            range_i = np.arange(0, (data_.shape[0] - 2 * self.border) // (self.patch_h - 2 * self.border)) * (
                self.patch_h - 2 * self.border)
            range_j = np.arange(0, (data_.shape[1] - 2 * self.border) // (self.patch_h - 2 * self.border)) * (
                self.patch_h - 2 * self.border)


            x_excess = np.mod(data_.shape[0] - 2 * self.border, self.patch_h - 2 * self.border)
            y_excess = np.mod(data_.shape[1] - 2 * self.border, self.patch_h - 2 * self.border)


            if not (x_excess == 0):
                if self.keep_edges:
                    range_i = np.append(range_i, (data_.shape[0] - self.patch_h))
                else:
                    print('{} pixels in x axis will be discarded'.format(x_excess))
            if not (y_excess == 0):
                if self.keep_edges:
                    range_j = np.append(range_j, (data_.shape[1] - self.patch_h))
                else:
                    print('{} pixels in y axis will be discarded'.format(y_excess))



            # nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)
            nr_patches = len(range_i)*len(range_j)

            print('{} Shapes Original = {}'.format(i, data_.shape))
            print('   Shapes Patched (hi-res) Dataset = {}'.format(
                [nr_patches, self.patch_h, self.patch_h, data_.shape[-1]]))

            self.dataset[i] = data_

            self.nr_patches[i] = nr_patches
            self.range_i[i] = range_i
            self.range_j[i] = range_j

    def get_inputs(self):
        return self.inputs_queue.get()

    def get_iter(self):
        while True:
            yield self.get_inputs()[0:2]


class PatchExtractor1:
    def __init__(self, dataset_low, dataset_high = None, batch_size=100, patch_l=16, max_queue_size=4, n_workers=8,
                 weights=None, is_random=True, border=None, scale=None, return_corner=False, keep_edges=True):

        self.dataset_l = dataset_low
        self.dataset_h = dataset_high
        self.is_random = is_random
        self.border = border
        self.scale = scale
        self.batch_size = batch_size
        self.weights = weights
        self.return_corner = return_corner
        self.keep_edges = keep_edges

        self.patch_l = patch_l
        self.patch_h = patch_l * scale

        if not isinstance(self.dataset_l, list):
            self.dataset_l = [self.dataset_l]

        if not self.is_random:
            self.compute_tile_ranges()
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
                self.inputs_queue.put(self.get_random_batch())
        else:

            while True:
                i = 0
                with self.lock:
                    for data_id in range(len(self.dataset_l)):
                        for ii in self.range_i[data_id]:
                            for jj in self.range_j[data_id]:
                                # print(i, ii, jj)
                                self.inputs_queue.put(self.get_patch_and_observe(xy_corner=(ii, jj), id=data_id)) # + (i,)
                                i += 1
                    print 'starting over Val set {}'.format(i)


    def get_patch_and_observe(self, xy_corner, id=0):

        if self.scale is None:
            scale = 1
        else:
            scale = self.scale
        x_l, y_l = xy_corner
        x_h, y_h = x_l*scale, y_l*scale


        if self.dataset_h is not None:
            label_patch = self.dataset_h[x_h:x_h + self.patch_h,
                          y_h:y_h+self.patch_h]
            assert label_patch.shape == (self.patch_h, self.patch_h, self.dataset_h.shape[-1],),\
                'Shapes: Dataset={} Patch={} xy_corner={}'.format(self.dataset_h.shape, label_patch.shape, (x_h,y_h))
        else:
            label_patch = None
        if self.scale is not None:
            # lr = []
            # for j, coord_ in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
            #     temp = observe_image(label_patch, coord_, scale=self.scale)
            #     lr.append(temp)
            features = self.dataset_l[id][x_l:x_l + self.patch_l,
                      y_l:y_l+self.patch_l]
            # features = np.stack(lr)
            if not self.is_random:
                label_patch = np.expand_dims(label_patch,axis=0)
                features = np.expand_dims(features,axis=0)
        else:
            features = None
        if self.return_corner:
            return features, label_patch, xy_corner
        else:
            return features, label_patch

    def get_random_batch(self):
        feature_sets = []
        label_sets = []

        if len(self.dataset_l) > 1:
            # We choose one of the datasets weighted on the relative number of pixels they have
            id = np.random.choice(len(self.dataset_l), p=self.weights)
        else:
            id = 0

        n_x, n_y = np.subtract(self.dataset_l[id].shape[0:2], self.patch_l)
        # Corner is always computed in low_res data
        ind = np.random.choice(n_x * n_y, int(self.batch_size), replace=False)

        for i in range(self.batch_size):
            corner_ = divmod(ind[i], n_y)
            feat_patch, label_patch = self.get_patch_and_observe(corner_, id)
            feature_sets.append(feat_patch)
            label_sets.append(label_patch)

        return (np.array(feature_sets), np.array(label_sets))

    def compute_tile_ranges(self):

        borders = (self.border, self.border)
        borders_h = (self.border*self.scale, self.border*self.scale)

        self.range_i = [None] * len(self.dataset_l)
        self.range_j = [None] * len(self.dataset_l)
        self.nr_patches = [None] * len(self.dataset_l)

        for i, data_ in enumerate(self.dataset_l):

            data_ = np.pad(data_, (borders, borders, (0, 0)), mode='symmetric')

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
            nr_patches = len(range_i)*len(range_j)

            print('{} Shapes Original = {}'.format(i, data_.shape))
            print('   Shapes Patched (low-res) Dataset = {}'.format(
                [nr_patches, self.patch_l, self.patch_l, data_.shape[-1]]))

            self.dataset_l[i] = data_

            self.nr_patches[i] = nr_patches
            self.range_i[i] = range_i
            self.range_j[i] = range_j

        self.dataset_h = np.pad(self.dataset_h, (borders_h, borders_h, (0, 0)), mode='symmetric') if self.dataset_h is not None else None

    def get_inputs(self):
        return self.inputs_queue.get()

    def get_iter(self):
        while True:
            yield self.get_inputs()[0:2]
