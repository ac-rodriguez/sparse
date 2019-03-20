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

from plots import check_dims, plot_rgb, plot_heatmap

from read_geoTiff import readHR, readS2
import gdal_processing as gp

IS_DEBUG=False
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


def read_labels(args, roi, roi_with_labels, is_HR=False):
    # if args.HR_file is not None:
    ref_scale=16 # 10m -> 0.625m
    sigma = ref_scale/np.pi
    if is_HR:
        ds_file = args.HR_file
        ref_scale = ref_scale//args.scale
        scale_lims = args.scale
    else:
        ds_file = os.path.join(os.path.dirname(args.LR_file), 'geotif', 'Band_B3.tif')
        scale_lims = 1

    print(' [*] Reading Labels {}'.format(os.path.basename(args.points)))

    ds = gdal.Open(ds_file)
    print(' [*] Reading complete Area')

    lims_H = gp.to_xy_box(roi, ds, enlarge=scale_lims)
    print(' [*] Reading labeled Area')

    lims_with_labels = gp.to_xy_box(roi_with_labels, ds, enlarge=scale_lims)

    labels = gp.rasterize_points_constrained(Input=args.points, refDataset=ds_file, lims=lims_H,
                                             lims_with_labels=lims_with_labels, up_scale=ref_scale,
                                             sigma=sigma, sq_kernel=args.sq_kernel)
    (xmin,ymin,xmax,ymax) = lims_with_labels
    xmin, xmax = xmin -lims_H[0], xmax - lims_H[0]
    ymin, ymax = ymin -lims_H[1], ymax - lims_H[1]
    return np.expand_dims(labels, axis=2), (xmin,ymin,xmax,ymax)


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

    def __init__(self, args, is_training, is_random_patches = True):
        '''Initialise a DataReader.

        Args:
          args: arguments from the train.py
        '''
        self.args = args
        self.two_ds = True
        self.is_training = is_training
        self.run_60 = False
        self.luminosity_scale = 9999.0
        self.batch = self.args.batch_size
        self.patch_l = self.args.patch_size
        self.patch_l_eval = self.args.patch_size_eval

        self.scale = self.args.scale
        self.gsd = str(10.0/self.scale)
        self.args.HR_file = os.path.join(self.args.HR_file,'3000_gsd{}.tif'.format(self.gsd))
        self.patch_h = self.args.patch_size * self.scale

        self.args.max_N = 1e5

        self.read_data(self.is_training)
        if self.is_training:
            if self.args.numpy_seed: np.random.seed(self.args.numpy_seed)
            if args.sigma_smooth:
                self.labels = ndimage.gaussian_filter(self.labels, sigma=args.sigma_smooth)

            self.n_channels = self.train[0].shape[-1]

            d2_after = self.args.unlabeled_after * self.batch
            self.patch_gen = PatchExtractor(dataset_low=self.train, dataset_high=self.train_h, label=self.labels,
                                            patch_l=self.patch_l, n_workers=4,max_queue_size=10, is_random=is_random_patches,
                                            scale=args.scale, max_N=args.train_patches, lims_with_labels=self.lims_labels,  patches_with_labels=self.args.patches_with_labels, d2_after=d2_after, two_ds=self.two_ds)

            # featl,datah = self.patch_gen.get_inputs()
            # plt.imshow(datah[...,0:3])
            # plt.imshow(featl[...,-1])
            # im = plot_rgb(featl, file='', return_img=True)
            # im.show()
            #
            # im = plot_rgb(feath, file='', return_img=True)
            # im.show()


            # print('Done')

            # self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
            #                                     patch_l=self.patch_l, n_workers=1, is_random=False, border=4,
            #                                     scale=args.scale)
            self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
                                                patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10, is_random=is_random_patches,border=4,
                                                scale=args.scale, max_N=args.val_patches, lims_with_labels=self.lims_labels_val, patches_with_labels=self.args.patches_with_labels, two_ds=self.two_ds)

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

            self.patch_gen = PatchExtractor(dataset_low=self.train, dataset_high=self.train_h, label=self.labels,
                                                 patch_l=self.patch_l, n_workers=1, is_random=False, border=4,
                                                 scale=args.scale, lims_with_labels=self.lims_labels, patches_with_labels=self.args.patches_with_labels, two_ds=True)

            self.n_channels = self.train[0].shape[-1]

            # self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=None, label=self.labels_val,
            #                                      patch_l=self.patch_l, n_workers=1, is_random=False, border=4,
            #                                      scale=args.scale)

            # for _ in range(10):
            #     feat,_ = self.patch_gen_test.get_inputs()
            #
            #     # # plt.imshow(label.squeeze())
            #     im = plot_rgb(feat,file ='', return_img=True)
            #     im.show()
            #
            # print('done')
            if False:
                nr_patches = self.patch_gen.nr_patches
                rgb_rec = np.empty(shape=[nr_patches, args.patch_size*args.scale, args.patch_size*args.scale,3])

                for idx in xrange(0, nr_patches):
                    # for idx in xrange(0,100):
                    lr, hr = self.patch_gen.get_inputs()

                    rgb_rec[idx] = hr[...,0:3]
                    # pred_c_rec[start:stop] = pc_


                    if idx % 1000 == 0:
                        print(idx)
                        # recompose
                border = 4
                ref_size = (self.train_h.shape[1],self.train_h.shape[0])
                print ref_size
                ## Recompose RGB
                import patches
                data_recomposed = patches.recompose_images(rgb_rec, size=ref_size, border=border*args.scale)
                # plots.plot_heatmap(data_recomposed, file=model_dir + '/pred_reg_recomposed')  # ,min=-1,max=1)
                plot_rgb(data_recomposed,file='data_recomposed')

    def non_random_patches(self):
        if self.is_training:
            self.patch_gen = PatchExtractor(dataset_low=self.train, dataset_high=self.train_h, label=self.labels,
                                            patch_l=self.patch_l, n_workers=1, is_random=False, border=4,
                                            scale=self.args.scale, lims_with_labels=self.lims_labels,  patches_with_labels=self.args.patches_with_labels,two_ds=True)

        self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
                                            patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                                            is_random=False, border=4,
                                            scale=self.args.scale, lims_with_labels=self.lims_labels_val,  patches_with_labels=self.args.patches_with_labels, two_ds=self.two_ds)

    def normalize(self, features, labels):

        features['feat_l'] -= self.mean_train
        features['feat_l'] /= self.std_train
        if self.two_ds:
            features['feat_lU'] -= self.mean_train
            features['feat_lU'] /= self.std_train
        return features, labels


    def add_spatial_masking(self, features, labels):

        a = tf.random.uniform([2],maxval=int(self.patch_l),dtype=tf.int32)
        size = int(self.patch_l / 2.0)
        int_ = lambda x: tf.cast(x, dtype=tf.int32)
        index = tf.constant(np.arange(0,self.patch_l), dtype=tf.int32)
        index_x = tf.logical_and(index > a[0],index < (a[0]+size))
        index_y = tf.logical_and(index > a[1],index < (a[1] + size))

        w = int_(tf.reshape(index_x, [-1,1,1])) * int_(tf.reshape(index_y, [1,-1,1]))

        labels = tf.where(w > 0,-1.0 * tf.ones_like(labels),labels)

        return features, labels
    def reorder_ds(self, data_l, data_h):
        n = self.n_channels
        if not self.two_ds:
            if self.is_HR_labels:
                return {'feat_l': data_l,
                        'feat_h': data_h[..., 0:3]}, tf.expand_dims(data_h[..., -1], axis=-1)
            else:
                return {'feat_l': data_l[..., 0:n],
                        'feat_h': data_h}, tf.expand_dims(data_l[..., -1], axis=-1)
        else:
            if self.is_HR_labels:
                return {'feat_l': data_l[...,:n],
                        'feat_h': data_h[..., 0:3],
                        'feat_lU': data_l[...,n:],
                        'feat_hU': data_h[..., 4:7],},\
                       tf.expand_dims(data_h[..., 3], axis=-1)
            else:
                return {'feat_l': data_l[..., 0:n],
                        'feat_h': data_h[...,0:3],
                        'feat_lU': data_l[..., (n+1):2*n+1],
                        'feat_hU': data_h[..., 3:]},\
                       tf.expand_dims(data_l[..., n], axis=-1)

    def read_data(self, is_training=True):
        self.is_HR_labels = self.args.is_hr_label
        if is_training:
            print('\n [*] Loading TRAIN data \n')
            self.train_h = readHR(self.args,
                                  roi_lon_lat=self.args.roi_lon_lat_tr) if self.args.HR_file is not None else None
            self.train = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)
            self.labels, self.lims_labels = read_labels(self.args, roi=self.args.roi_lon_lat_tr,
                                      roi_with_labels=self.args.roi_lon_lat_tr_lb, is_HR=self.is_HR_labels)

            print('\n [*] Loading VALIDATION data \n')
            self.val_h = readHR(self.args,
                                roi_lon_lat=self.args.roi_lon_lat_val) if self.args.HR_file is not None else None
            self.val = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_val)
            self.labels_val,self.lims_labels_val = read_labels(self.args, roi=self.args.roi_lon_lat_val,
                                          roi_with_labels=self.args.roi_lon_lat_val_lb, is_HR=self.is_HR_labels)

            # sum_train = np.expand_dims(self.train.sum(axis=(0, 1)),axis=0)
            # self.N_pixels = self.train.shape[0] * self.train.shape[1]

            self.mean_train = self.train.mean(axis=(0, 1))
            self.max_dens = self.labels.max()
            self.std_train = self.train.std(axis=(0, 1))
            if IS_DEBUG:
                f1 = lambda x: (np.where(x == -1, x, x * (2.0 / self.max_dens)) if self.is_HR_labels else x)
                plt_reg = lambda x, file: plot_heatmap(f1(x), file=file, min=-1, max=2.0, cmap='jet')
                plt_reg(self.labels, self.args.model_dir + '/train_reg_label')
                plot_rgb(self.train_h, file=self.args.model_dir + '/train_HR', reorder=False, percentiles=(0, 100))

                plt_reg(self.labels_val, self.args.model_dir + '/val_reg_label')
                plot_rgb(self.val_h, file=self.args.model_dir + '/val_HR', reorder=False, percentiles=(0, 100))

                # sys.exit(0)

            if self.args.is_empty_aerial:
                self.train[(self.labels == -1)[..., 0], :] = 2000.0
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
                self.train = self.train[0:x_shapes, 0:y_shapes, :]
                if self.is_HR_labels:
                    self.labels = self.labels[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                else:
                    self.labels = self.labels[0:x_shapes, 0:y_shapes]

                print('\tAfter:\tLow:({}x{})\tHigh:({}x{})'.format(self.train.shape[0], self.train.shape[1],
                                                                   self.train_h.shape[0], self.train_h.shape[1]))

                x_shapes, y_shapes = self.compute_shapes(dset_h=self.val_h, dset_l=self.val, scale=scale)

                print(' Val shapes: \n\tBefore:\tLow:({}x{})\tHigh:({}x{})'.format(self.val.shape[0], self.val.shape[1],
                                                                                   self.val_h.shape[0],
                                                                                   self.val_h.shape[1]))

                # Reduce data to the enlarged 10m pixels
                self.val_h = self.val_h[0:int(scale * x_shapes), 0:int(scale * y_shapes), :]
                self.val = self.val[0:int(x_shapes), 0:int(y_shapes), :]
                if self.is_HR_labels:
                    self.labels_val = self.labels_val[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                else:
                    self.labels_val = self.labels_val[0:x_shapes, 0:y_shapes]

                print('\tAfter:\tLow:({}x{})\tHigh:({}x{})'.format(self.val.shape[0], self.val.shape[1],
                                                                   self.val_h.shape[0], self.val_h.shape[1]))

            if self.args.is_padding:
                a = self.patch_l - 1
                self.train =  np.pad(self.train, ((a,a),(a,a),(0,0)), mode='constant', constant_values=0.0)
                b = a * scale
                self.train_h = np.pad(self.train_h, ((b,b),(b,b),(0,0)), mode='constant', constant_values = 0.0) if self.train_h is not None else self.train_h
                c = b if self.is_HR_labels else a
                self.labels = np.pad(self.labels, ((c,c),(c,c),(0,0)), mode='constant', constant_values = -1.0)

                print('Padded datasets with low ={}, high={} with 0.0'.format(a,b))

        else:
            self.train_h = readHR(self.args,
                                  roi_lon_lat=self.args.roi_lon_lat_tr) if self.args.HR_file is not None else None
            self.train = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)
            self.labels, _ = read_labels(self.args, roi=self.args.roi_lon_lat_tr,
                                      roi_with_labels=self.args.roi_lon_lat_tr_lb, is_HR=self.is_HR_labels)

            self.nb_bands = self.train.shape[-1]
            self.mean_train = np.zeros(self.nb_bands)
            self.std_train = np.ones(self.nb_bands)

            scale = self.args.scale
            if self.train_h is not None:
                x_shapes, y_shapes = self.compute_shapes(dset_h=self.train_h, dset_l=self.train, scale=scale)

                print(
                    ' Predict shapes: \n\tBefore:\tLow:({}x{})\tHigh:({}x{})'.format(self.train.shape[0],
                                                                                   self.train.shape[1],
                                                                                   self.train_h.shape[0],
                                                                                   self.train_h.shape[1]))
                # Reduce data to the enlarged 10m pixels
                self.train_h = self.train_h[0:int(scale * x_shapes), 0:int(scale * y_shapes), :]
                self.train = self.train[0:x_shapes, 0:y_shapes, :]
                if self.is_HR_labels:
                    self.labels = self.labels[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                else:
                    self.labels = self.labels[0:x_shapes, 0:y_shapes]

                print('\tAfter:\tLow:({}x{})\tHigh:({}x{})'.format(self.train.shape[0], self.train.shape[1],
                                                                   self.train_h.shape[0], self.train_h.shape[1]))


        if False:
            print(' [*] Loading data for Prediction ')
            # self.train_h = readHR(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)
            self.train = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)

            self.labels,_ = read_labels(self.args, roi=self.args.roi_lon_lat_tr,
                                      roi_with_labels=self.args.roi_lon_lat_tr_lb) if self.args.points is not None else None
            self.train_h = readHR(self.args,
                                  roi_lon_lat=self.args.roi_lon_lat_tr) if self.args.HR_file is not None else None

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
    @staticmethod
    def compute_shapes(scale, dset_h, dset_l):

        enlarge = lambda x: int(x / scale) * scale
        x_h, y_h, _ = map(enlarge, dset_h.shape)
        x_l, y_l, _ = dset_l.shape
        x_shape, y_shape = min(int(x_h / scale), x_l), min(int(y_h / scale), y_l)
        return x_shape, y_shape

    def init_constants_normalization(self):
        tf.Variable(self.mean_train.astype(np.float32), name='mean_train', trainable=False, validate_shape=True,
                    expected_shape=tf.shape([self.n_channels]))

        tf.Variable(self.std_train.astype(np.float32), name='std_train', trainable=False,
                    expected_shape=tf.shape([self.n_channels]))
        tf.Variable(self.max_dens.astype(np.float32), name='max_dens', trainable=False,
                    expected_shape=tf.shape([1]))

        tf.constant(self.mean_train.astype(np.float32), name='mean_train_k')

        tf.constant(self.std_train.astype(np.float32), name='std_train_k')
        tf.constant(self.max_dens.astype(np.float32), name='max_dens_k')
    def input_fn(self, is_train=True):
        # np.random.seed(99)
        self.init_constants_normalization()

        if is_train:
            gen_func = self.patch_gen.get_iter
            patch_l, patch_h = self.patch_l, self.patch_h
        else:
            gen_func = self.patch_gen_val.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)

        multiplier = 2 if self.two_ds else 1

        if self.is_HR_labels:
            n_low = self.n_channels *multiplier
            n_high = 4 *multiplier
        else:
            n_low = (self.n_channels + 1)*multiplier
            n_high = 3 *multiplier
        ds = tf.data.Dataset.from_generator(
            gen_func, (tf.float32, tf.float32),
            (
                tf.TensorShape([patch_l, patch_l,
                                n_low]),
                tf.TensorShape([patch_h, patch_h,
                                n_high])
            ))

        ds = ds.map(self.reorder_ds, num_parallel_calls=6)
        if is_train and self.args.is_masking:
            ds = ds.map(self.add_spatial_masking)
        ds = ds.batch(self.batch).map(self.normalize, num_parallel_calls=6)

        ds = ds.prefetch(buffer_size=self.args.batch_size*2)

        return ds

    def get_input_fn(self):
        input_fn = None
        if self.is_training:
            input_fn = partial(self.input_fn, is_train=True)
        input_fn_val = partial(self.input_fn, is_train=False)

        return input_fn, input_fn_val

    # def input_fn_test(self):
    #     self.init_constants_normalization()
    #     gen_func = self.patch_gen_test.get_iter
    #
    #     if self.labels is not None:
    #         ds = tf.data.Dataset.from_generator(
    #             gen_func, (tf.float32, tf.float32),
    #             (
    #                 tf.TensorShape([self.patch_l, self.patch_l,
    #                                 self.n_channels]),
    #                 tf.TensorShape([self.patch_h, self.patch_h,
    #                                 1])
    #             ))
    #         ds = ds.map(self.reorder_ds)
    #     else:
    #         ds = tf.data.Dataset.from_generator(
    #             gen_func, (tf.float32, tf.float32),
    #             (
    #                 tf.TensorShape([self.patch_l, self.patch_l,
    #                                 self.n_channels]),
    #                 tf.TensorShape([None])
    #             ))
    #         ds = ds.map(self.normalize)
    #
    #     ds = ds.batch(self.batch).prefetch(buffer_size=10)
    #
    #     return ds


class PatchExtractor:
    def __init__(self, dataset_low, dataset_high, label, patch_l=16, max_queue_size=4, n_workers=1, is_random=True,
                 border=None, scale=None, return_corner=False, keep_edges=True, max_N=5e4, lims_with_labels = None, patches_with_labels= 0.1, d2_after=0, two_ds=True):
        self.two_ds = two_ds

        self.d_l = dataset_low
        self.d_h = dataset_high
        self.label = label
        self.patches_with_labels = patches_with_labels
        self.lims_lab = lims_with_labels
        self.d2_after = d2_after
        if IS_DEBUG:
            self.d_l1 = np.zeros_like(self.d_l)
            self.label_1 = np.zeros_like(self.label)
        self.is_HR_label = not (self.d_l.shape[0:2] == self.label.shape[0:2])

        self.is_random = is_random
        self.border = border
        self.scale = scale
        self.nr_patches = max_N
        self.return_corner = return_corner
        self.keep_edges = keep_edges

        self.patch_l = patch_l
        assert self.patch_l <= self.d_l.shape[0] and self.patch_l <= self.d_l.shape[1], \
            ' patch of size {} is bigger than ds_l {}'.format(self.patch_l, self.d_l.shape)

        self.patch_h = patch_l * scale
        self.patch_lab = self.patch_h if self.is_HR_label else self.patch_l


        if self.border is not None:
            assert self.patch_l > self.border * 2
            self.border_lab = self.border * self.scale if self.is_HR_label else self.border
        if not self.is_random:
            self.compute_tile_ranges()

        else:

            n_x, self.n_y = np.subtract(self.d_l.shape[0:2], self.patch_l)
            print('Max N random patches = {}'.format(n_x*self.n_y))
            # Corner is always computed in low_res data
            max_patches = min(self.nr_patches, n_x*self.n_y)
            print('Extracted random patches = {}'.format(max_patches))


            if self.two_ds:
                buffer_size = 0
                size_label_ind = max_patches

            else:
                size_label_ind = int(max_patches*self.patches_with_labels)

                # Patches with ground truth
                buffer_size = self.patch_l //2

            if self.is_HR_label:
                ymin, xmin, ymax, xmax = map(lambda x: x // self.scale, self.lims_lab)
            else:
                ymin, xmin, ymax, xmax = self.lims_lab
            print  self.lims_lab
            xmax,ymax = min(xmax, n_x), min(ymax, self.n_y)
            xmin,ymin = max(xmin-buffer_size,0),max(ymin-buffer_size,0)
            area_labels = (xmax-xmin+buffer_size,ymax-ymin+buffer_size)
            # area_labels = (50,50)

            indices = np.random.choice(area_labels[0]*area_labels[1],size=min(area_labels[0]*area_labels[1],size_label_ind),replace=False)
            dx, dy, n_y_lab = max(0,xmin -buffer_size//2), max(0,ymin -buffer_size//2), area_labels[1]

            coords = np.array(map(lambda x: divmod(x,n_y_lab),indices))

            coords1 = coords + (dx,dy)

            indices = coords1[:,0]*self.n_y + coords1[:,1]
            assert np.max(indices) < (n_x*self.n_y), (np.max(indices), (n_x*self.n_y))

            if self.two_ds:
                self.indices1 = indices
                self.indices2 = np.random.choice(n_x * self.n_y, size=len(self.indices1), replace=False)

                print(' labeled and unlabeled data are always fed within a batch')

            else:

                # Corner is always computed in low_res data
                indices1 = np.random.choice(n_x * self.n_y, size=int(max_patches)-len(indices), replace=False)

                self.indices = np.concatenate((indices,indices1))
                self.shuffled_indices = False
                self.len_labels = len(indices)
                self.len_labelsALL = len(self.indices)

                print(' {}/{} are patches with GT'.format(len(indices),len(self.indices)))

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
                            if self.two_ds:
                                patches = self.get_patches(xy_corner=(ii, jj))
                                patches = np.concatenate((patches[0], patches[0]), axis=-1), np.concatenate(
                                    (patches[1], patches[1]), axis=-1)
                                self.inputs_queue.put(patches)
                            else:
                                self.inputs_queue.put(self.get_patches(xy_corner=(ii, jj)))  # + (i,)
                            i += 1
                    print 'starting over Val set {}'.format(i)

    def get_patch_corner(self,data, x,y,size):
        if data is not None:
            patch = data[x:x + size, y:y+size]

            assert patch.shape == (size, size, data.shape[-1],), \
                'Shapes: Dataset={} Patch={} xy_corner={}'.format(data.shape, patch.shape, (x, y))
        else:
            patch=None
        return patch
    def get_patches(self, xy_corner):

        if self.scale is None:
            scale = 1
        else:
            scale = self.scale
        x_l, y_l = map(int, xy_corner)
        x_h, y_h = x_l * scale, y_l * scale
        x_lab,y_lab = (x_h,y_h) if self.is_HR_label else (x_l,y_l)

        patch_h = self.get_patch_corner(self.d_h,x_h,y_h,self.patch_h)
        label = self.get_patch_corner(self.label,x_lab,y_lab,self.patch_lab)
        # if self.scale is not None:
        patch_l = self.get_patch_corner(self.d_l, x_l, y_l, self.patch_l)

        if IS_DEBUG and label is not None:
            self.label_1[x_lab:x_lab + self.patch_lab,
                y_lab:y_lab + self.patch_lab] = label

        if IS_DEBUG and patch_l is not None:
                self.d_l1[x_l:x_l + self.patch_l,
                      y_l:y_l + self.patch_l]=patch_l

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


        if not self.two_ds:
            with self.lock:
                if self.rand_ind < self.d2_after:
                    ind = self.indices[np.mod(self.rand_ind,self.len_labels)]
                else:
                    if not self.shuffled_indices:
                        np.random.shuffle(self.indices)
                        self.shuffled_indices = True
                        print(' Shuffling and addind unlabeled data')
                    ind = self.indices[np.mod(self.rand_ind,self.len_labelsALL)]

                # print ' rand_index={}'.format(self.rand_ind)
                self.rand_ind+=1
            # ind = 50
            corner_ = divmod(ind, self.n_y)

            return self.get_patches(corner_)
        else:
            with self.lock:
                ind1 = self.indices1[np.mod(self.rand_ind, len(self.indices1))]
                ind2 = self.indices2[np.mod(self.rand_ind, len(self.indices2))]

                # print ' rand_index={}'.format(self.rand_ind)
                self.rand_ind += 1
                # ind = 50
            corner1_ = divmod(ind1, self.n_y)
            corner2_ = divmod(ind2, self.n_y)
            patches1 = self.get_patches(corner1_)
            patches2 = self.get_patches(corner2_)
            patches = np.concatenate((patches1[0],patches2[0]),axis=-1) , np.concatenate((patches1[1],patches2[1]),axis=-1)
            return patches


    def compute_tile_ranges(self):

        borders = (self.border, self.border)
        borders_h = (self.border * self.scale, self.border * self.scale)
        borders_lab = (self.border_lab, self.border_lab)
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

        self.label = np.pad(self.label, (borders_lab, borders_lab, (0, 0)),
                            mode='symmetric') if self.label is not None else None


        if IS_DEBUG:
            self.d_l1 = np.zeros_like(self.d_l)
            self.label_1 = np.zeros_like(self.label)
    def get_inputs(self):
        return self.inputs_queue.get()

    def get_iter(self):
        while True:
            yield self.get_inputs()[0:2]
