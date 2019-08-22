import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize, downscale_local_mean
from itertools import compress
from functools import partial

import plots
import read_geoTiff as geotif
from patch_extractor import PatchExtractor, PatchWraper


def interpPatches(image_20lr, ref_shape=None, squeeze=False, scale=None, mode='reflect'):
    image_20lr = plots.check_dims(image_20lr)
    N, w, h, ch = image_20lr.shape

    if scale is not None:
        ref_shape = (int(w * scale), int(h * scale))
    image_20lr = np.rollaxis(image_20lr, -1, 1)

    data20_interp = np.zeros(((N, ch) + ref_shape)).astype(np.float32)
    for k in range(N):
        for w in range(ch):
            data20_interp[k, w] = resize(image_20lr[k, w] / 65000, ref_shape, mode=mode) * 65000  # bicubic

    data20_interp = np.rollaxis(data20_interp, 1, 4)
    if squeeze:
        data20_interp = np.squeeze(data20_interp, axis=0)

    return data20_interp


def downscale(img, scale):
    scale = int(scale)
    h, w, ch = img.shape
    ref_shape = (h // scale, w // scale, ch)
    imout = np.zeros(ref_shape)
    for c in range(ch):
        imout[..., c] = resize(img[..., c], ref_shape[0:2], anti_aliasing=True,
                               anti_aliasing_sigma=float(scale) / np.pi, mode='reflect')

    return imout


def read_and_upsample_sen2(data_file, args, roi_lon_lat, mask_out_dict=None):
    if data_file is None:
        return None
        # if 'LR_file' in args:
        #     LR_file = args.LR_file
        #     if ';' in LR_file:
        #         LR_file = LR_file.split(';')[0]
        # else:
        #     LR_file = args.data_dir
    if mask_out_dict is not None:
        is_get_SCL = 'SCL' in mask_out_dict.keys()
    else:
        is_get_SCL = False
    if 'USER' in data_file:
        data10, data20 = geotif.readS2_old(args, roi_lon_lat, data_file=data_file)
    else:
        data10, data20 = geotif.readS2(args, roi_lon_lat, data_file=data_file, is_get_SCL=is_get_SCL)
    if data10 is None:
        return None
    if mask_out_dict is not None:
        mask = np.zeros_like(data20[...,0], dtype=np.bool)
        for key, val in mask_out_dict.items():
            if 'CLD' == key:
                mask = np.logical_or(mask,data20[...,-2]>val)
            elif 'SCL' == key:
                for item in val:
                    mask = np.logical_or(mask, data20[..., -1] == item)
            elif '20m' == key:
                missing = (data20[..., 0:6] == val).any(axis=-1)
                mask = np.logical_or(mask, missing)

        mask = interpPatches(mask, data10.shape[0:2], squeeze=True, mode='edge') > 0.5
        data20 = data20[...,0:-1]  # Dropping the SCL band

    data20 = interpPatches(data20, data10.shape[0:2], squeeze=True)

    data = np.concatenate((data10, data20), axis=2)
    if mask_out_dict is not None:
        # mask = np.repeat(mask,data.shape[-1], axis=2)
        data[mask.squeeze()] = np.nan
    size = np.prod(data.shape) * data.itemsize

    print('{:.2f} GB loaded in memory'.format(size / 1e9))
    return data


class DataReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, args, is_training, is_random_patches=True):
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
        self.gsd = str(10.0 / self.scale)

        self.is_upsample = self.args.is_upsample_LR
        if "HR_file" in self.args and not self.args.HR_file is None:
            self.args.HR_file = os.path.join(self.args.HR_file, '3000_gsd{}.tif'.format(self.gsd))
        else:
            self.args.HR_file = None
        for i in self.args.tr+self.args.val+self.args.test:
            if i['hr'] is not None and not i['hr'].endswith('.tif'):
                i['hr'] = os.path.join(i['hr'], '3000_gsd{}.tif'.format(self.gsd))
        self.patch_h = self.args.patch_size * self.scale

        self.args.max_N = 1e5

        if self.is_training:
            self.read_train_data()
            if self.args.numpy_seed: np.random.seed(self.args.numpy_seed)
            if args.sigma_smooth:
                self.labels = ndimage.gaussian_filter(self.labels, sigma=args.sigma_smooth)

            self.n_channels = self.train[0].shape[-1]
            d2_after = self.args.unlabeled_after * self.batch_tr
            self.single_gen = []
            if not args.is_total_patches_datasets:
                train_patches = [args.train_patches] * len(self.train)
            else:
                pixels_ = [x.shape[0] * x.shape[1] for x in self.train]
                train_patches = [args.train_patches*x // np.sum(pixels_) for x in pixels_]

                # Adding rounding down patches left out in the first dataset
                train_patches[0] = train_patches[0] + args.train_patches - np.sum(train_patches)

            for id_ in range(len(self.train)):
                self.single_gen.append(
                    PatchExtractor(dataset_low=self.train[id_], dataset_high=self.train_h[id_], label=self.labels[id_],
                                   patch_l=self.patch_l, n_workers=4, max_queue_size=10, is_random=is_random_patches,
                                   scale=args.scale, max_N=train_patches[id_], lims_with_labels=self.lims_labels[id_],
                                   patches_with_labels=self.args.patches_with_labels, d2_after=d2_after,
                                   two_ds=self.two_ds,
                                   unlab=self.unlab))
            self.patch_gen = PatchWraper(self.single_gen, n_workers=4, max_queue_size=10)

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
            # val_random = True if self.args.dataset == 'vaihingen' else False
            # if val_random:
            self.single_gen_val = []
            self.single_gen_rand_val = []
            # val_dset_ = -1 # TODO implement several datasets for val dset
            for val_dset_ in range(len(self.val)):
                self.single_gen_val.append(
                    PatchExtractor(dataset_low=self.val[val_dset_], dataset_high=self.val_h[val_dset_],
                                                             label=self.labels_val[val_dset_],
                                                             patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                                                             is_random=False, border=4,
                                                             scale=self.scale, lims_with_labels=self.lims_labels_val[val_dset_],
                                                             patches_with_labels=self.args.patches_with_labels,
                                                             two_ds=self.two_ds))
                if self.single_gen_val[-1].nr_patches*10 >= args.val_patches:

                    self.single_gen_rand_val.append(
                        PatchExtractor(dataset_low=self.val[val_dset_], dataset_high=self.val_h[val_dset_],
                                                             label=self.labels_val[val_dset_],
                                                             patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                                                             is_random=True, border=4,
                                                             scale=args.scale, max_N=args.val_patches,
                                                             lims_with_labels=self.lims_labels_val[val_dset_],
                                                             patches_with_labels=self.args.patches_with_labels,
                                                             two_ds=self.two_ds))
                else:
                    print(f' Complete val dataset has {self.single_gen_val[-1].nr_patches} and it is used as random sub-sample too...')

                    self.single_gen_rand_val.append(self.single_gen_val[-1])

            # self.patch_gen_val_complete = PatchWraper(self.single_gen_val, n_workers=4, max_queue_size=10)
            self.patch_gen_val_rand = PatchWraper(self.single_gen_rand_val, n_workers=4, max_queue_size=10)
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
            self.prepare_test_data()
            # self.read_test_data()
            # self.two_ds = False
            # self.patch_gen_test = PatchExtractor(dataset_low=self.test[0], dataset_high=self.test_h[0],
            #                                      label=self.labels_test[0],
            #                                      patch_l=self.patch_l_eval, n_workers=1, is_random=False, border=4,
            #                                      scale=self.scale, lims_with_labels=self.lims_labels_test[0],
            #                                      patches_with_labels=self.args.patches_with_labels, two_ds=self.two_ds)

            # for _ in range(10):
            #     feat,_ = self.patch_gen_test.get_inputs()
            #
            #     # # plt.imshow(label.squeeze())
            #     im = plot_rgb(feat,file ='', return_img=True)
            #     im.show()
            #
            # print('done')

    def read_data_pairs(self, path_dict, upsample_lr=True, is_vaihingen=False, ref_scale=1, mask_out_dict=None):

        if is_vaihingen:
            train_h = geotif.readHR(data_file=path_dict['hr'], roi_lon_lat=None, scale=self.scale)

            train = downscale(train_h.copy(), ref_scale)
            train_h = downscale(train_h, ref_scale // self.scale)

            labels, lim_labels = geotif.read_labels_semseg(self.args, sem_file=path_dict['sem'],
                                                           dsm_file=path_dict['dsm'],
                                                           is_HR=self.is_HR_labels, ref_scale=ref_scale)
        else:

            train_h = geotif.readHR(data_file=path_dict['hr'], roi_lon_lat=path_dict['roi'],
                                    scale=self.scale)
            if self.args.is_noS2:
                assert train_h is not None
                train = downscale(train_h.copy(), scale=self.scale)
            else:
                train = read_and_upsample_sen2(data_file=path_dict['lr'], args=self.args, roi_lon_lat=path_dict['roi'], mask_out_dict=mask_out_dict)

            if train_h is None and upsample_lr:
                train_h = interpPatches(train, scale=self.scale)
                train_h = np.clip(train_h[..., (2, 3, 1)] / 4000, 0, 1).squeeze()

            labels, lim_labels = geotif.read_labels(self.args, shp_file=path_dict['gt'], roi=path_dict['roi'], roi_with_labels=path_dict['roi_lb'],
                                           is_HR=self.is_HR_labels, ref_lr=path_dict['lr'], ref_hr=path_dict['hr'])
            if labels is not None:
                if 'age' in self.args.dataset:
                    print(f' Ages: {np.unique(labels)}')
                else:
                    print(f' Densities percentiles 10,20,50,70,90 \n {np.percentile(labels, q=[0.1,0.2,0.5,0.7,0.9])}')
                    if 'palmcoco' in self.args.dataset:
                        if 'labels/palm' in path_dict['gt']:
                            labels = np.concatenate((labels,np.zeros_like(labels)), axis=-1)

                            print('Palm Object - class 1')
                        elif 'labels/coco/' in path_dict['gt']:
                            labels = np.concatenate((np.zeros_like(labels),labels),axis=-1)
                            print('Coco Object - class 2')
                        elif 'labels/coconutSHP/' in path_dict['gt']:
                            labels = np.concatenate((labels == 6, labels == 2), axis=-1) ## Palm is code 6 and coco code 2
                            labels = np.int32(labels) * 0.8 # without density Gt we just define it as 0.8 trees/pixel if there is a tree
                            print('Coco and Palm Object')


        return train, train_h, labels, lim_labels

    def read_train_data(self):
        self.is_HR_labels = self.args.is_hr_label
        self.unlab = None

        print('\n [*] Loading TRAIN data \n')
        is_vaihingen = 'vaihingen' in self.args.dataset

        ref_scale = self.scale if is_vaihingen else 16

        self.train_h = []
        self.train = []
        self.labels = []
        self.lims_labels = []
        is_valid = []

        for tr_ in self.args.tr:
            train, train_h, labels, lim_labels = self.read_data_pairs(tr_, is_vaihingen=is_vaihingen,ref_scale=ref_scale, upsample_lr=self.is_upsample)
            is_valid.append(train is not None)
            if is_valid[-1]:
                self.train.append(train)
                self.train_h.append(train_h)
                self.labels.append(labels)
                self.lims_labels.append(lim_labels)
        self.train_info = list(compress(self.args.tr,is_valid))

        if self.args.unlabeled_data is not None:
            self.unlab = read_and_upsample_sen2(data_file=self.args.unlabeled_data, args=self.args,
                                                roi_lon_lat=self.args.roi_lon_lat_unlab)
        self.n_classes = self.labels[0].shape[-1]

        print('\n [*] Loading VALIDATION data \n')

        self.val_h = []
        self.val = []
        self.labels_val = []
        self.lims_labels_val = []
        is_valid = []
        maskout = {'CLD':10, # if Cloud prob is higher than 10%
                   'SCL':[3,11], # if SCL is equal to cloud or cloud shadow
                   '20m':0} # if all 20m bands are 0
        for val_ in self.args.val:
            val, val_h, labels_val, lim_labels_val = self.read_data_pairs(val_, is_vaihingen=is_vaihingen,ref_scale=ref_scale, upsample_lr=self.is_upsample, mask_out_dict=maskout)
            is_valid.append(val is not None)
            if is_valid[-1]:
                # mask out clouds
                # val[val[..., -1] > 50] = np.nan

                self.val.append(val)
                self.val_h.append(val_h)
                self.labels_val.append(labels_val)
                self.lims_labels_val.append(lim_labels_val)
        self.val_info = list(compress(self.args.val,is_valid))
        self.val_tilenames = [x['tilename'] for x in self.val_info]

        # sum_train = np.expand_dims(self.train.sum(axis=(0, 1)),axis=0)
        # self.N_pixels = self.train.shape[0] * self.train.shape[1]

        self.mean_train = self.train[0].mean(axis=(0, 1))
        self.max_dens = self.labels[0].max()
        self.std_train = self.train[0].std(axis=(0, 1))

        if self.args.is_empty_aerial:
            self.train[(self.labels == -1)[..., 0], :] = 2000.0
        print(str(self.mean_train))
        self.nb_bands = self.train[0].shape[-1]

        scale = self.scale
        for i_ in range(len(self.train)):
            if self.train_h[i_] is not None:
                self.train_h[i_],self.train[i_],self.labels[i_] = self.correct_shapes(
                    self.train_h[i_],self.train[i_],self.labels[i_], scale,type='Train',is_hr_lab=self.is_HR_labels)

        for i_ in range(len(self.val)):

            if self.val_h[i_] is not None:
                self.val_h[i_], self.val[i_], self.labels_val[i_] = self.correct_shapes(
                    self.val_h[i_], self.val[i_], self.labels_val[i_], scale, type='Val', is_hr_lab=self.is_HR_labels)

        if self.args.save_arrays:
            max_ = 20.0 if 'palmage' in self.args.dataset else 2.0
            f1 = lambda x: (np.where(x == -1, x, x * (max_ / self.max_dens)) if self.is_HR_labels else x)
            plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, min=-1, max=max_, cmap='viridis')
            if 'palmcoco' in self.args.dataset or 'cococomplete' in self.args.dataset:
                for tile in set(self.val_tilenames):
                    index = [tile == x for x in self.val_tilenames]
                    val = list(compress(self.val, index))
                    lab_val = list(compress(self.labels_val, index))

                    shapes = set([x.shape for x in val])
                    for i,s in enumerate(shapes):
                        val_sameshape = [x for x in val if s == x.shape]
                        val_sameshape = np.stack(val_sameshape, axis=-1)
                        val_sameshape = np.nanmedian(val_sameshape, axis=-1)
                        plots.plot_rgb(val_sameshape, file=self.args.model_dir + f'/val_LR_{tile}_{i}')

                        np.save(self.args.model_dir + f'/val_LR_{tile}_{i}', val_sameshape)
                        lab_sameshape = [x for x in lab_val if s[0:2] == x.shape[0:2]]
                        for j in range(self.n_classes):
                            plt_reg(lab_sameshape[0][..., j], self.args.model_dir + f'/val_reg_label{tile}_class{j}_{i}')
                        np.save(self.args.model_dir + f'/val_reg_label_{tile}_{i}', lab_sameshape[0])


            else:
                for i_ in range(len(self.val)):
                    if 'vaihingen' in self.args.dataset:
                        plots.plot_heatmap(self.labels_val[i_][..., 1], file=self.args.model_dir + f'/val_reg_label{i_}', min=0,
                                           percentiles=(0, 100))
                        plots.plot_labels(self.labels_val[i_][..., 0], file=self.args.model_dir + f'/val_sem_label{i_}')
                        plots.plot_rgb(self.val[i_], file=self.args.model_dir + f'/val_LR{i_}', reorder=False, normalize=False)

                        plots.plot_rgb(self.val_h[i_], file=self.args.model_dir + f'/val_HR{i_}', reorder=False, normalize=False)
                        np.save(self.args.model_dir + f'/val_LR{i_}', self.val[i_])
                        np.save(self.args.model_dir + f'/val_HR{i_}', self.val_h[i_])
                        np.save(self.args.model_dir + f'/val_reg_label{i_}', self.labels_val[i_])
                    else:
                        for j in range(self.n_classes):
                            plt_reg(self.labels_val[i_][...,j], self.args.model_dir + f'/val_reg_label{i_}_class{j}')
                        plots.plot_rgb(self.val[i_], file=self.args.model_dir + f'/val_LR{i_}')

                        if self.val_h[i_] is not None:
                            plots.plot_rgb(self.val_h[i_], file=self.args.model_dir+f'/val_HR{i_}', reorder=False, normalize=False)
                            np.save(self.args.model_dir + f'/val_HR{i_}', self.val_h[i_])

                        np.save(self.args.model_dir + f'/val_LR{i_}', self.val[i_])
                        np.save(self.args.model_dir + f'/val_reg_label{i_}', self.labels_val[i_])
            # for i_ in range(len(self.train)):
            #     if 'vaihingen' in self.args.dataset:
            #         plots.plot_heatmap(self.labels[i_][..., 1], file=self.args.model_dir + f'/train_reg_label{i_}', min=0,
            #                            percentiles=(0, 100))
            #         plots.plot_labels(self.labels[i_][..., 0], file=self.args.model_dir + f'/train_sem_label{i_}')
            #         plots.plot_rgb(self.train[i_], file=self.args.model_dir + f'/train_LR{i_}', reorder=False, normalize=False)
            #
            #         plots.plot_rgb(self.train_h[i_], file=self.args.model_dir + f'/train_HR{i_}', reorder=False, normalize=False)
            #         np.save(self.args.model_dir + f'/train_LR{i_}', self.train[i_])
            #         np.save(self.args.model_dir + f'/train_HR{i_}', self.train_h[i_])
            #         np.save(self.args.model_dir + f'/train_reg_label{i_}', self.labels[i_])
            #     else:
            #
            #         plt_reg(self.labels[i_], self.args.model_dir + f'/train_reg_label{i_}')
            #         plots.plot_rgb(self.train[i_], file=self.args.model_dir + f'/train_LR{i_}')
            #
            #         plots.plot_rgb(self.train_h[i_], file=self.args.model_dir+f'/train_HR{i_}', reorder=False, normalize=False)
            #         np.save(self.args.model_dir + f'/train_LR{i_}', self.train[i_])
            #         np.save(self.args.model_dir + f'/train_HR{i_}', self.train_h[i_])
            #         np.save(self.args.model_dir + f'/train_reg_label{i_}', self.labels[i_])
            # sys.exit(0)

        if self.args.is_padding:
            a = self.patch_l - 1
            self.train = np.pad(self.train, ((a, a), (a, a), (0, 0)), mode='constant', constant_values=0.0)
            b = a * scale
            self.train_h = np.pad(self.train_h, ((b, b), (b, b), (0, 0)), mode='constant',
                                  constant_values=0.0) if self.train_h is not None else self.train_h
            c = b if self.is_HR_labels else a
            self.labels = np.pad(self.labels, ((c, c), (c, c), (0, 0)), mode='constant', constant_values=-1.0)

            print('Padded datasets with low ={}, high={} with 0.0'.format(a, b))

    def read_test_data(self):

        print('\n [*] Loading TEST data \n')

        self.is_HR_labels = self.args.is_hr_label
        is_vaihingen = 'vaihingen' in self.args.dataset

        ref_scale = self.scale if is_vaihingen else 16

        self.test_h = []
        self.test = []
        self.labels_test = []
        self.lims_labels_test = []
        is_valid = []
        maskout = {'CLD':10, # if Cloud prob is higher than 10%
                   'SCL':[3,11,6], # if SCL is equal to cloud shadow, snow or water
                   '20m':0} # if all 20m bands are 0

        for test_ in self.args.test:

            test, test_h, labels_test, lim_labels_test = self.read_data_pairs(test_,
                                                                              upsample_lr=self.is_upsample,
                                                                              is_vaihingen=is_vaihingen,
                                                                              ref_scale=ref_scale, mask_out_dict=maskout)
            is_valid.append(test is not None)
            if is_valid[-1]:
                # test[test[..., -2] > 50] = np.nan
                # test[test[..., -1] == 50] = np.nan

                self.test.append(test)
                self.test_h.append(test_h)
                self.labels_test.append(labels_test)
                self.lims_labels_test.append(lim_labels_test)
        self.test_info = list(compress(self.args.test,is_valid))
        assert len(self.test_info) > 0, 'no valid datasets were left'
        self.test_tilenames = [x['tilename'] for x in self.test_info]

        scale = self.scale

        for i_ in range(len(self.test)):
            if self.test_h[i_] is not None:
                self.test_h[i_], self.test[i_], self.labels_test[i_] = self.correct_shapes(
                    self.test_h[i_], self.test[i_], self.labels_test[i_], scale, type='Test', is_hr_lab=self.is_HR_labels)

        if not hasattr(self,'n_classes'):
            if self.labels_test[0] is not None:
                self.n_classes = self.labels_test[0].shape[-1]
            elif 'palmcoco' in self.args.model_dir:
                self.n_classes = 2
            else:
                self.n_classes = 1
        self.n_channels = self.test[0].shape[-1]

        if self.args.save_arrays:
            f1 = lambda x: (np.where(x == -1, x, x * (2.0 / self.max_dens)) if self.is_HR_labels else x)
            plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, min=-1, max=2.0, cmap='viridis')
            if 'palmcoco' in self.args.dataset or 'cococomplete' in self.args.dataset:
                for tile in set(self.test_tilenames):
                    index = [tile == x for x in self.test_tilenames]
                    test = list(compress(self.test, index))
                    lab_test = list(compress(self.labels_test, index))
                    if lab_test[0] is not None:
                        for j in range(self.n_classes):
                            plt_reg(lab_test[0][..., j], self.args.model_dir + f'/test_reg_label{tile}_class{j}')

                    test = np.stack(test, axis=-1)
                    test = np.nanmedian(test, axis=-1)
                    plots.plot_rgb(test, file=self.args.model_dir + f'/test_LR_{tile}')

                    np.save(self.args.model_dir + f'/test_LR_{tile}', test)
                    np.save(self.args.model_dir + f'/test_reg_label_{tile}', lab_test[0])
            else:

                for i_ in range(len(self.test)):
                    if self.labels_test[i_] is not None:
                        plt_reg(self.labels_test[i_], self.args.model_dir + f'/test_reg_label{i_}')
                    if self.test_h[i_] is not None:
                        plots.plot_rgb(self.test_h[i_], file=self.args.model_dir + f'/test_HR{i_}', reorder=False, percentiles=(0, 100))
                    plots.plot_rgb(self.test[i_], file=self.args.model_dir + f'/test_S2{i_}')

    def correct_shapes(self,high, low, label, scale, type, is_hr_lab):
        x_shapes, y_shapes = self.compute_shapes(dset_h=high, dset_l=low, scale=scale)
        print(f' {type} shapes: \n\tBefore:\tLow:{low.shape[0:2]}\tHigh:{high.shape[0:2]}')

        # Reduce data to the enlarged 10m pixels
        high = high[0:int(scale * x_shapes), 0:int(scale * y_shapes), :]
        low = low[0:x_shapes, 0:y_shapes, :]
        if is_hr_lab:
            label = label[0:int(scale * x_shapes), 0:int(scale * y_shapes)]
        else:
            label = label[0:x_shapes, 0:y_shapes]
        print(f'\tAfter:\tLow:{low.shape[0:2]}\tHigh:{high.shape[0:2]}')
        return high, low, label

    @staticmethod
    def compute_shapes(scale, dset_h, dset_l):

        enlarge = lambda x: int(x / scale) * scale
        x_h, y_h, _ = map(enlarge, dset_h.shape)
        x_l, y_l, _ = dset_l.shape
        x_shape, y_shape = min(int(x_h / scale), x_l), min(int(y_h / scale), y_l)
        return x_shape, y_shape

    def prepare_test_data(self):

        self.read_test_data()
        self.two_ds = False

        self.single_gen_test = []


        for test_dset_ in range(len(self.test)):
            self.single_gen_test.append(
                PatchExtractor(dataset_low=self.test[test_dset_], dataset_high=self.test_h[test_dset_],
                               label=self.labels_test[test_dset_],
                               patch_l=self.patch_l_eval, n_workers=4, max_queue_size=10,
                               is_random=False, border=4,
                               scale=self.scale, lims_with_labels=self.lims_labels_test[test_dset_],
                               patches_with_labels=self.args.patches_with_labels,
                               two_ds=self.two_ds))



    def get_input_test(self, is_restart=False, as_list=False):
        if not as_list:
            if is_restart:
                self.patch_gen_test.define_queues()

            return partial(self.input_fn, type='test')
        else:
            list_fn = []
            for id_, d_ in enumerate(self.single_gen_test):
                if is_restart:
                    d_.define_queues()
                list_fn.append(partial(self.input_fn,type=f'test_complete-{id_}'))
            return list_fn

    def get_input_test_old(self):  #
        self.init_constants_normalization()

        gen_func = self.patch_gen_test.get_iter_test
        patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
        batch = self.batch_eval
        multiplier = 2 if self.two_ds else 1
        self.n_channels_lab = 2 if 'vaihingen' in self.args.dataset else 1
        n_lab = self.n_channels_lab
        if self.labels_test[0] is not None:
            if self.is_HR_labels:
                n_low = self.n_channels * multiplier
                n_high = (3 + n_lab) * multiplier
            else:
                n_low = (self.n_channels + n_lab) * multiplier
                n_high = 3 * multiplier

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

            ds = ds.prefetch(buffer_size=batch * 2)
        else:
            ds = tf.data.Dataset.from_generator(
                gen_func, tf.float32,
                tf.TensorShape([patch_l, patch_l, self.n_channels]))
            if batch is None:
                batch = self.args.batch_size
            normalize = lambda x: (x - self.mean_train) / self.std_train
            ds = ds.batch(batch).map(normalize, num_parallel_calls=6)

            ds = ds.prefetch(buffer_size=batch * 2)

        return ds

    def get_input_val(self, is_restart=False, as_list=False):
        if not as_list:
            if is_restart:
                self.patch_gen_val_complete.define_queues()

            return partial(self.input_fn, type='val_complete')
        else:
            list_fn = []
            for id_, d_ in enumerate(self.single_gen_val):
                if is_restart:
                    d_.define_queues()
                list_fn.append(partial(self.input_fn,type=f'val_complete-{id_}'))
            return list_fn

    def normalize(self, features, labels):

        if isinstance(features,dict):
            features['feat_l'] -= self.mean_train
            features['feat_l'] /= self.std_train
            if self.two_ds:
                features['feat_lU'] -= self.mean_train
                features['feat_lU'] /= self.std_train
        else:
            features -= self.mean_train
            features /= self.std_train
        return features, labels

    def add_spatial_masking(self, features, labels):

        a = tf.random.uniform([2], maxval=int(self.patch_l), dtype=tf.int32)
        size = int(self.patch_l / 2.0)
        int_ = lambda x: tf.cast(x, dtype=tf.int32)
        index = tf.constant(np.arange(0, self.patch_l), dtype=tf.int32)
        index_x = tf.logical_and(index > a[0], index < (a[0] + size))
        index_y = tf.logical_and(index > a[1], index < (a[1] + size))

        w = int_(tf.reshape(index_x, [-1, 1, 1])) * int_(tf.reshape(index_y, [1, -1, 1]))

        labels = tf.where(w > 0, -1.0 * tf.ones_like(labels), labels)

        return features, labels

    # def reorder_ds(self, data_l, data_h):
    def reorder_ds(self,data):
        if isinstance(data,tuple):
            data_l, data_h = data
        else:
            data_l,data_h = data,None

        n = self.n_channels
        n_c = self.n_classes
        if 'vaihingen' in self.args.dataset:
            if self.is_HR_labels:
                return {'feat_l': data_l[..., :n],
                        'feat_h': data_h[..., 0:3],
                        'feat_lU': data_l[..., n:],
                        'feat_hU': data_h[..., 5:8], }, \
                       data_h[..., 3:5]
            else:
                return {'feat_l': data_l[..., 0:n],
                        'feat_h': data_h[..., 0:3],
                        'feat_lU': data_l[..., (n + 2):2 * n + 2],
                        'feat_hU': data_h[..., 3:]}, \
                       data_l[..., n:n + 2]

        elif not self.args.is_upsample_LR:
            if self.two_ds:
                return {'feat_l':data_l[..., 0:n],'feat_lU':data_l[..., (n + n_c):2 * n + n_c]}, data_l[..., n:n+n_c]
            else:
                return data_l[..., 0:n], data_l[...,n:n+n_c]
        else:

            if not self.two_ds:
                if self.is_HR_labels:
                    return {'feat_l': data_l,
                            'feat_h': data_h[..., 0:3]}, tf.expand_dims(data_h[..., -n_c:], axis=-1)
                else:
                    return {'feat_l': data_l[..., 0:n],
                            'feat_h': data_h}, tf.expand_dims(data_l[..., -n_c], axis=-1)
            else:
                assert n_c == 1, 'not implemented for more classes'
                if self.is_HR_labels:
                    return {'feat_l': data_l[..., :n],
                            'feat_h': data_h[..., 0:3],
                            'feat_lU': data_l[..., n:],
                            'feat_hU': data_h[..., 4:7], }, \
                           tf.expand_dims(data_h[..., 3], axis=-1)
                else:
                    return {'feat_l': data_l[..., 0:n],
                            'feat_h': data_h[..., 0:3],
                            'feat_lU': data_l[..., (n + 1):2 * n + 1],
                            'feat_hU': data_h[..., 3:]}, \
                           tf.expand_dims(data_l[..., n], axis=-1)

    def init_constants_normalization(self):

        try:
            tf.Variable(self.mean_train.astype(np.float32), name='mean_train', trainable=False, validate_shape=True,
                        expected_shape=tf.shape([self.n_channels]))

            tf.Variable(self.std_train.astype(np.float32), name='std_train', trainable=False,
                        expected_shape=tf.shape([self.n_channels]))
            tf.Variable(self.max_dens.astype(np.float32), name='max_dens', trainable=False,
                        expected_shape=tf.shape([1]))

            tf.constant(self.mean_train.astype(np.float32), name='mean_train_k')

            tf.constant(self.std_train.astype(np.float32), name='std_train_k')
            tf.constant(self.max_dens.astype(np.float32), name='max_dens_k')
        except AttributeError:

            self.mean_train = tf.Variable(np.zeros(self.n_channels, dtype=np.float32), name='mean_train',
                                          trainable=False, validate_shape=True,
                                          expected_shape=tf.shape([self.n_channels]))

            self.std_train = tf.Variable(np.ones(self.n_channels, dtype=np.float32), name='std_train', trainable=False,
                                         expected_shape=tf.shape([self.n_channels]))

    def input_fn(self, type='train'):
        # np.random.seed(99)
        self.init_constants_normalization()

        if type == 'train':
            gen_func = self.patch_gen.get_iter
            patch_l, patch_h = self.patch_l, self.patch_h
            batch = self.batch_tr
            is_onlyLR = self.patch_gen.is_onlyLR
        elif type == 'val':
            gen_func = self.patch_gen_val_rand.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
            is_onlyLR = self.patch_gen_val_rand.is_onlyLR
        elif type == 'val_complete':
            gen_func = self.patch_gen_val_complete.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
            is_onlyLR = self.patch_gen_val_complete.is_onlyLR
        elif 'val_complete' in type:
            type,id_ = type.split('-')
            id_ = int(id_)
            gen_func = self.single_gen_val[id_].get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
            is_onlyLR = self.single_gen_val[id_].is_onlyLR
        elif type == 'test':
            gen_func = self.patch_gen_test.get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
            is_onlyLR = self.patch_gen_test.is_onlyLR
        elif 'test_complete' in type:
            type,id_ = type.split('-')
            id_ = int(id_)
            gen_func = self.single_gen_test[id_].get_iter
            patch_l, patch_h = self.patch_l_eval, int(self.patch_l_eval * self.scale)
            batch = self.batch_eval
            is_onlyLR = self.single_gen_test[id_].is_onlyLR
        else:
            sys.exit(1)
        multiplier = 2 if self.two_ds else 1
        self.n_channels_lab = 2 if 'vaihingen' in self.args.dataset or 'palmcoco' in self.args.dataset else 1
        n_lab = self.n_channels_lab
        if 'test' in type and self.labels_test[0] is None:
            ds = tf.data.Dataset.from_generator(
                gen_func, tf.float32,
                tf.TensorShape([patch_l, patch_l, self.n_channels]))
            if batch is None:
                batch = self.args.batch_size
            normalize = lambda x: (x - self.mean_train) / self.std_train
            ds = ds.batch(batch).map(normalize, num_parallel_calls=6)

            ds = ds.prefetch(buffer_size=batch * 2)
            return ds

        if self.is_HR_labels:
            n_low = self.n_channels * multiplier
            n_high = (3 + n_lab) * multiplier
        else:
            n_low = (self.n_channels + n_lab) * multiplier
            n_high = 3 * multiplier
        if is_onlyLR:
            ds = tf.data.Dataset.from_generator(
                gen_func, (tf.float32),
                (
                    tf.TensorShape([patch_l, patch_l,
                                    n_low])
                ))
        else:
            ds = tf.data.Dataset.from_generator(
                gen_func, (tf.float32, tf.float32),
                (
                    tf.TensorShape([patch_l, patch_l,
                                    n_low]),
                    tf.TensorShape([patch_h, patch_h,
                                    n_high])
                ))
        if type == 'train':
            ds = ds.shuffle(buffer_size=batch * 5)
        ds = ds.map(self.reorder_ds, num_parallel_calls=6)
        if type == 'train' and self.args.is_masking:
            ds = ds.map(self.add_spatial_masking)
        ds = ds.batch(batch).map(self.normalize, num_parallel_calls=6)

        ds = ds.prefetch(buffer_size=batch * 2)

        return ds

    def get_input_fn(self, is_val_random=False):
        val_type = 'val' if is_val_random else 'val_complete'
        input_fn = None
        if self.is_training:
            input_fn = partial(self.input_fn, type='train')
        input_fn_val = partial(self.input_fn, type=val_type)

        return input_fn, input_fn_val
