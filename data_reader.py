import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize, downscale_local_mean

from functools import partial

import plots
import read_geoTiff as geotif
from patch_extractor import PatchExtractor


def interpPatches(image_20lr, ref_shape=None, squeeze=False, scale=None):
    image_20lr = plots.check_dims(image_20lr)
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
def downscale(img, scale):
    scale = int(scale)
    h, w, ch = img.shape
    ref_shape = (h// scale,w//scale,ch)
    imout = np.zeros(ref_shape)
    for c in range(ch):
        imout[...,c] = resize(img[...,c], ref_shape[0:2], anti_aliasing=True, anti_aliasing_sigma=float(scale)/np.pi, mode='reflect')

    return imout


def read_and_upsample_sen2(args, roi_lon_lat, LR_file=None):
    if LR_file is None:
        if 'LR_file' in args:
            LR_file = args.LR_file
        else:
            LR_file = args.data_dir
    if 'USER' in LR_file:
        data10, data20 = geotif.readS2_old(args, roi_lon_lat, data_file=LR_file)
    else:
        data10, data20 = geotif.readS2(args, roi_lon_lat,data_file=LR_file)
    data20 = interpPatches(data20, data10.shape[0:2], squeeze=True)

    data = np.concatenate((data10, data20), axis=2)

    size = np.prod(data.shape) * data.itemsize

    print('{:.2f} GB loaded in memory'.format(size / 1e9))
    return data




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
        self.batch_tr = self.args.batch_size
        self.batch_eval = self.args.batch_size_eval

        self.patch_l = self.args.patch_size
        self.patch_l_eval = self.args.patch_size_eval

        self.scale = self.args.scale
        self.gsd = str(10.0/self.scale)

        if "HR_file" in self.args:
            self.args.HR_file = os.path.join(self.args.HR_file, '3000_gsd{}.tif'.format(self.gsd))
        else:
            self.args.HR_file = None

        self.patch_h = self.args.patch_size * self.scale

        self.args.max_N = 1e5

        if self.is_training:
            self.read_train_data()
            if self.args.numpy_seed: np.random.seed(self.args.numpy_seed)
            if args.sigma_smooth:
                self.labels = ndimage.gaussian_filter(self.labels, sigma=args.sigma_smooth)

            self.n_channels = self.train[0].shape[-1]

            d2_after = self.args.unlabeled_after * self.batch_tr
            self.patch_gen = PatchExtractor(dataset_low=self.train, dataset_high=self.train_h, label=self.labels,
                                            patch_l=self.patch_l, n_workers=4,max_queue_size=10, is_random=is_random_patches,
                                            scale=args.scale, max_N=args.train_patches, lims_with_labels=self.lims_labels,
                                            patches_with_labels=self.args.patches_with_labels, d2_after=d2_after, two_ds=self.two_ds,
                                            unlab=self.unlab)

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
            val_random = True if self.args.dataset == 'vaihingen' else False
            if val_random:
                self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
                                                    patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10, is_random=True,border=4,
                                                    scale=args.scale, max_N=args.val_patches, lims_with_labels=self.lims_labels_val, patches_with_labels=self.args.patches_with_labels, two_ds=self.two_ds)
            else:
                self.patch_gen_val = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h,
                                                    label=self.labels_val,
                                                    patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                                                    is_random=False, border=4,
                                                    scale=self.args.scale, lims_with_labels=self.lims_labels_val,
                                                    patches_with_labels=self.args.patches_with_labels,
                                                    two_ds=self.two_ds)

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
            self.read_test_data()
            self.n_channels = self.test[0].shape[-1]
            self.two_ds = False
            self.patch_gen_test = PatchExtractor(dataset_low=self.test, dataset_high=self.test_h, label=self.labels_test,
                                                 patch_l=self.patch_l_eval, n_workers=1, is_random=False, border=4,
                                                 scale=self.args.scale, lims_with_labels=self.lims_labels_test,
                                                 patches_with_labels=self.args.patches_with_labels, two_ds=self.two_ds)

            # for _ in range(10):
            #     feat,_ = self.patch_gen_test.get_inputs()
            #
            #     # # plt.imshow(label.squeeze())
            #     im = plot_rgb(feat,file ='', return_img=True)
            #     im.show()
            #
            # print('done')

    def read_train_data(self):
        self.is_HR_labels = self.args.is_hr_label
        self.unlab = None

        print('\n [*] Loading TRAIN data \n')

        if 'vaihingen' in self.args.dataset:
            # LR space reference is always 16 ds. from original HR image
            self.train_h = geotif.readHR(self.args, roi_lon_lat=None, data_file=self.args.top_train)
            ref_scale = 16
            self.train = downscale(self.train_h.copy(),ref_scale)
            self.train_h = downscale(self.train_h,ref_scale//self.args.scale)
            self.labels,self.lims_labels = geotif.read_labels_semseg(self.args, sem_file=self.args.sem_train, dsm_file=self.args.dsm_train, is_HR=self.is_HR_labels, ref_scale=ref_scale)
        else:
            self.train_h = geotif.readHR(self.args,
                                         roi_lon_lat=self.args.roi_lon_lat_tr) if self.args.HR_file is not None else None
            if self.args.is_noS2:
                self.train = downscale(self.train_h.copy(), scale=self.args.scale)
            else:
                self.train = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_tr)

            self.labels, self.lims_labels = geotif.read_labels(self.args, roi=self.args.roi_lon_lat_tr,
                                      roi_with_labels=self.args.roi_lon_lat_tr_lb, is_HR=self.is_HR_labels)
            if self.args.unlabeled_data is not None:
                self.unlab = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_unlab,LR_file=self.args.unlabeled_data)

        print('\n [*] Loading VALIDATION data \n')
        if 'vaihingen' in self.args.dataset:
            ref_scale = 16
            if self.args.top_train == self.args.top_val:
                self.val_h = self.train_h
                self.val = self.train
            else:
                self.val_h = geotif.readHR(self.args, roi_lon_lat=None, data_file=self.args.top_val)
                self.val = downscale(self.val_h.copy(), scale=ref_scale)
                self.val_h = downscale(self.val_h, scale=ref_scale//self.args.scale)

            self.labels_val,self.lims_labels_val = geotif.read_labels_semseg(self.args, sem_file=self.args.sem_val, dsm_file=self.args.dsm_val, is_HR=self.is_HR_labels, ref_scale=ref_scale)

        else:
            self.val_h = geotif.readHR(self.args,
                                roi_lon_lat=self.args.roi_lon_lat_val) if self.args.HR_file is not None else None
            if self.args.is_noS2:
                self.val = downscale(self.val_h,scale=self.args.scale)
            else:
                self.val = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_val)

            self.labels_val,self.lims_labels_val = geotif.read_labels(self.args, roi=self.args.roi_lon_lat_val,
                                          roi_with_labels=self.args.roi_lon_lat_val_lb, is_HR=self.is_HR_labels)

        # sum_train = np.expand_dims(self.train.sum(axis=(0, 1)),axis=0)
        # self.N_pixels = self.train.shape[0] * self.train.shape[1]

        self.mean_train = self.train.mean(axis=(0, 1))
        self.max_dens = self.labels.max()
        self.std_train = self.train.std(axis=(0, 1))


        if self.args.is_empty_aerial:
            self.train[(self.labels == -1)[..., 0], :] = 2000.0
        print(str(self.mean_train))
        self.nb_bands = self.train.shape[-1]

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

        if self.args.save_arrays:
            f1 = lambda x: (np.where(x == -1, x, x * (2.0 / self.max_dens)) if self.is_HR_labels else x)
            plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, min=-1, max=2.0, cmap='viridis')

            if 'vaihingen' in self.args.dataset:
                plots.plot_heatmap(self.labels_val[...,1], file=self.args.model_dir + '/val_reg_label', min=0,percentiles=(0,100))
                plots.plot_labels(self.labels_val[...,0], file=self.args.model_dir + '/val_sem_label')
                plots.plot_rgb(self.val, file=self.args.model_dir + '/val_S2', reorder=False,normalize=False)

            else:
                plt_reg(self.labels_val, self.args.model_dir + '/val_reg_label')
                plots.plot_rgb(self.val, file=self.args.model_dir + '/val_S2')

            plots.plot_rgb(self.val_h, file=self.args.model_dir + '/val_HR', reorder=False, normalize=False)
            np.save(self.args.model_dir + '/val_S2', self.val)
            np.save(self.args.model_dir + '/val_HR', self.val_h)
            np.save(self.args.model_dir + '/val_reg_label', self.labels_val)
            # sys.exit(0)

        if self.args.is_padding:
            a = self.patch_l - 1
            self.train =  np.pad(self.train, ((a,a),(a,a),(0,0)), mode='constant', constant_values=0.0)
            b = a * scale
            self.train_h = np.pad(self.train_h, ((b,b),(b,b),(0,0)), mode='constant', constant_values = 0.0) if self.train_h is not None else self.train_h
            c = b if self.is_HR_labels else a
            self.labels = np.pad(self.labels, ((c,c),(c,c),(0,0)), mode='constant', constant_values = -1.0)

            print('Padded datasets with low ={}, high={} with 0.0'.format(a,b))
    def read_test_data(self):
        self.is_HR_labels = self.args.is_hr_label
        if 'vaihingen' in self.args.dataset:
            # LR space reference is always 16 ds. from original HR image
            self.test_h = geotif.readHR(self.args, roi_lon_lat=None, data_file=self.args.top_test)
            ref_scale = 16
            self.test = downscale(self.test_h.copy(),ref_scale)
            self.test_h = downscale(self.test_h,ref_scale//self.args.scale)
            self.labels_test,self.lims_labels_test = geotif.read_labels_semseg(self.args, sem_file=self.args.sem_test, dsm_file=self.args.dsm_test, is_HR=self.is_HR_labels, ref_scale=ref_scale)
        else:

            if not "roi_lon_lat_test" in self.args:
                self.args.roi_lon_lat_test = None
            self.test_h = geotif.readHR(self.args,
                                 roi_lon_lat=self.args.roi_lon_lat_test) if self.args.HR_file is not None else None
            if self.args.is_noS2:
                self.test = downscale(self.test_h,scale=self.args.scale)
            else:
                self.test = read_and_upsample_sen2(self.args, roi_lon_lat=self.args.roi_lon_lat_test)
            if "points" in self.args:
                self.labels_test, self.lims_labels_test = geotif.read_labels(self.args, roi=self.args.roi_lon_lat_test,
                                          roi_with_labels=self.args.roi_lon_lat_test, is_HR=self.is_HR_labels)
            else:
                self.labels_test, self.lims_labels_test = None, None

        scale = self.args.scale
        if self.test_h is not None:
            x_shapes, y_shapes = self.compute_shapes(dset_h=self.test_h, dset_l=self.test, scale=scale)
            print(
                ' Test shapes: \n\tBefore:\tLow:({}x{})\tHigh:({}x{})'.format(self.test.shape[0], self.test.shape[1],
                                                                              self.test_h.shape[0],
                                                                              self.test_h.shape[1]))
            self.test = self.test[0:int(x_shapes), 0:int(y_shapes), :]
            self.test_h = self.test_h[0:int(scale * x_shapes), 0:int(scale * y_shapes), :]

            if self.labels_test:
                if self.is_HR_labels:
                    self.labels_test = self.labels_test[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
                else:
                    self.labels_test = self.labels_test[0:x_shapes, 0:y_shapes]

            print('\tAfter:\tData:({}x{})\tLabels:({}x{})'.format(self.test.shape[0], self.test.shape[1],
                                                                  self.test_h.shape[0], self.test_h.shape[1]))


        if self.args.save_arrays:
            f1 = lambda x: (np.where(x == -1, x, x * (2.0 / self.max_dens)) if self.is_HR_labels else x)
            plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, min=-1, max=2.0, cmap='viridis')

            if self.labels_test is not None:
                plt_reg(self.labels_test, self.args.model_dir + '/test_reg_label')
            if self.test_h is not None:
                plots.plot_rgb(self.test_h, file=self.args.model_dir + '/test_HR', reorder=False, percentiles=(0, 100))
            plots.plot_rgb(self.test, file=self.args.model_dir + '/test_S2')

    @staticmethod
    def compute_shapes(scale, dset_h, dset_l):

        enlarge = lambda x: int(x / scale) * scale
        x_h, y_h, _ = map(enlarge, dset_h.shape)
        x_l, y_l, _ = dset_l.shape
        x_shape, y_shape = min(int(x_h / scale), x_l), min(int(y_h / scale), y_l)
        return x_shape, y_shape

    def prepare_test_data(self):

        self.read_test_data()

        self.patch_gen_test = PatchExtractor(dataset_low=self.test, dataset_high=self.test_h,
                                            label=self.labels_test,
                                            patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                                            is_random=False, border=4,
                                            scale=self.args.scale, lims_with_labels=self.lims_labels_test,
                                            patches_with_labels=self.args.patches_with_labels,
                                            two_ds=self.two_ds)

    def get_input_test(self): #

        gen_func = self.patch_gen_test.get_iter_test
        patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
        batch = self.batch_eval
        multiplier = 2 if self.two_ds else 1
        self.n_channels_lab = 2 if 'vaihingen' in self.args.dataset else 1
        n_lab = self.n_channels_lab
        if self.labels_test:
            if self.is_HR_labels:
                n_low = self.n_channels *multiplier
                n_high = (3+n_lab) *multiplier
            else:
                n_low = (self.n_channels + n_lab)*multiplier
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


            ds = ds.batch(batch).map(self.normalize, num_parallel_calls=6)

            ds = ds.prefetch(buffer_size=batch*2)
        else:
            ds = tf.data.Dataset.from_generator(
                gen_func, tf.float32,
                tf.TensorShape([patch_l, patch_l,self.n_channels]))
            if batch is None:
                batch = self.args.batch_size
            ds = ds.batch(batch) #.map(self.normalize, num_parallel_calls=6)

            ds = ds.prefetch(buffer_size=batch * 2)

        return ds


    def get_input_val(self):
        self.patch_gen_val1 = PatchExtractor(dataset_low=self.val, dataset_high=self.val_h, label=self.labels_val,
                                            patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                                            is_random=False, border=4,
                                            scale=self.args.scale, lims_with_labels=self.lims_labels_val,
                                            patches_with_labels=self.args.patches_with_labels, two_ds=self.two_ds)

        return partial(self.input_fn, type='val1')

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
        if 'vaihingen' in self.args.dataset:
            if self.is_HR_labels:
                return {'feat_l': data_l[...,:n],
                        'feat_h': data_h[..., 0:3],
                        'feat_lU': data_l[...,n:],
                        'feat_hU': data_h[..., 5:8],},\
                       data_h[..., 3:5]
            else:
                return {'feat_l': data_l[..., 0:n],
                        'feat_h': data_h[...,0:3],
                        'feat_lU': data_l[..., (n+2):2*n+2],
                        'feat_hU': data_h[..., 3:]},\
                       data_l[..., n:n+2]

        else:

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
    def input_fn(self, type='train'):
        # np.random.seed(99)
        self.init_constants_normalization()

        if type == 'train':
            gen_func = self.patch_gen.get_iter
            patch_l, patch_h = self.patch_l, self.patch_h
            batch = self.batch_tr
        elif type == 'val':
            gen_func = self.patch_gen_val.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
        elif type == 'val1':
            gen_func = self.patch_gen_val1.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
        elif type == 'test':
            gen_func = self.patch_gen_test.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
        else:
            sys.exit(1)
        multiplier = 2 if self.two_ds else 1
        self.n_channels_lab = 2 if 'vaihingen' in self.args.dataset else 1
        n_lab = self.n_channels_lab
        if self.is_HR_labels:
            n_low = self.n_channels *multiplier
            n_high = (3+n_lab) *multiplier
        else:
            n_low = (self.n_channels + n_lab)*multiplier
            n_high = 3 *multiplier
        ds = tf.data.Dataset.from_generator(
            gen_func, (tf.float32, tf.float32),
            (
                tf.TensorShape([patch_l, patch_l,
                                n_low]),
                tf.TensorShape([patch_h, patch_h,
                                n_high])
            ))
        if type == 'train':
            ds = ds.shuffle(buffer_size=batch*5)
        ds = ds.map(self.reorder_ds, num_parallel_calls=6)
        if type == 'train' and self.args.is_masking:
            ds = ds.map(self.add_spatial_masking)
        ds = ds.batch(batch).map(self.normalize, num_parallel_calls=6)

        ds = ds.prefetch(buffer_size=batch*2)

        return ds

    def get_input_fn(self):
        input_fn = None
        if self.is_training:
            input_fn = partial(self.input_fn, type='train')
        input_fn_val = partial(self.input_fn, type='val')

        return input_fn, input_fn_val


