import os, sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize, downscale_local_mean
from itertools import compress
from functools import partial
import ray
import json
import pickle
import tqdm
import threading
import utils.plots as plots
import utils.read_geoTiff as geotif
from utils.patch_extractor import PatchExtractor, PatchWraper
from utils import gdal_processing as gp
from osgeo import gdal


USE_RAY = False
IS_VERBOSE_DATALOAD = False
is_save_or_load_file = True

def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            return func
        return dec(func)
    return decorator

@conditional_decorator(ray.remote, USE_RAY)
def f_read(tr_,dict_id, mask_dict, read_data_pairs_func):
    id_lab = get_id_lab(tr_)

    train, train_h, labels, lim_labels = read_data_pairs_func(tr_, mask_out_dict=mask_dict, is_load_lab=False)

    labels = dict_id[id_lab][2]
    lim_labels = dict_id[id_lab][3]
    is_valid_ = train is not None
    return (train,train_h, labels,lim_labels,is_valid_)


from contextlib import redirect_stdout

def no_verbose(func):
    def decorated_func(*args, **kwargs):
        with open(os.devnull, 'w') as void:
            with redirect_stdout(void):
                return func(*args, **kwargs)
    return decorated_func


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


def read_and_upsample_sen2(data_file, args, roi_lon_lat, mask_out_dict=None,is_skip_if_masked=True):
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
        if np.mean(mask) > 0.5:
            if is_skip_if_masked:
                print(f" [!] {np.mean(mask)*100:.2f}% of data is masked, skipping it...")
                return None
            else:
                print(f" [!] {np.mean(mask)*100:.2f}% of data is masked")

        mask = interpPatches(mask, data10.shape[0:2], squeeze=True, mode='edge') > 0.5
        data20 = data20[...,0:-1]  # Dropping the SCL band

    data20 = interpPatches(data20, data10.shape[0:2], squeeze=True)

    data = np.concatenate((data10, data20), axis=2)
    if mask_out_dict is not None:
        # mask = np.repeat(mask,data.shape[-1], axis=2)
        data[mask.squeeze()] = np.nan
    size = np.prod(data.shape) * data.itemsize

    print('{:.2f} MB loaded in memory'.format(size / 1e6))
    return data


def check_path(list_in, basepath_):
    list_out = []
    add_path = lambda x: basepath_+'/barry_palm/'+x.split('barry_palm',1)[-1] if x is not None else x
    for i_ in list_in:
        i_['gt'] = add_path(i_['gt'])
        i_['lr'] = add_path(i_['lr'])
        list_out.append(i_)
    return list_out


def get_id_lab(x):
    if isinstance(x['roi'], tuple):
        x['roi'] = ','.join([str(i) for i in x['roi']])
    return '_'.join([x['gt'], x['roi']])

class DataReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, args, datatype, is_random_patches=True):
        '''Initialise a DataReader.

        Args:
          args: arguments from the train.py
        '''
        self.args = args
        self.two_ds = False
        self.datatype = datatype
        assert datatype in ['trainval','val','test'], datatype
        self.is_training = datatype == 'trainval'
        self.run_60 = False
        self.luminosity_scale = 9999.0
        self.batch_tr = self.args.batch_size
        self.batch_eval = self.args.batch_size_eval
        self.n_workers = self.args.n_workers

        self.patch_l = self.args.patch_size
        self.patch_l_eval = self.args.patch_size_eval

        self.scale = self.args.scale
        self.gsd = str(10.0 / self.scale)

        self.is_upsample = False

        self.args.HR_file = None
        
        self.is_HR_labels = False
        self.unlab = None

        self.patch_h = self.args.patch_size * self.scale

        self.args.max_N = 1e5

        if self.datatype == 'trainval':
            self.read_train_data()
            size = [np.prod(x.shape) * x.itemsize for x in self.train]
            print('TRAIN: {:.2f} MB loaded in memory'.format(np.sum(size) / 1e6))

            self.read_val_data()
            if self.args.numpy_seed: np.random.seed(self.args.numpy_seed)
            if args.sigma_smooth:
                self.labels = ndimage.gaussian_filter(self.labels, sigma=args.sigma_smooth)

            self.n_channels = self.train[0].shape[-1]
            # d2_after = self.args.unlabeled_after * self.batch_tr
            self.single_gen = []
            if not args.is_total_patches_datasets:
                train_patches = [args.train_patches] * len(self.train)
            else:
                pixels_ = [x.shape[0] * x.shape[1] for x in self.train]
                train_patches = [args.train_patches*x // np.sum(pixels_) for x in pixels_]

                # Adding rounding down patches left out in the first dataset
                train_patches[0] = train_patches[0] + args.train_patches - np.sum(train_patches)
            tbar = tqdm.trange(len(self.train), disable=False)
            for id_ in tbar:
                self.single_gen.append(
                    PatchExtractor(dataset_low=self.train[id_], dataset_high=self.train_h[id_], label=self.labels[id_],
                                   patch_l=self.patch_l, n_workers=self.n_workers, max_queue_size=10, is_random=is_random_patches,
                                   scale=args.scale, max_N=train_patches[id_], lims_with_labels=self.lims_labels[id_],
                                   patches_with_labels=self.args.patches_with_labels,
                                   two_ds=self.two_ds,
                                   unlab=self.unlab, use_location=args.is_use_location, is_use_queues=False))
                tbar.set_description(f'active_threads {threading.active_count()}')
            self.patch_gen = PatchWraper(self.single_gen, n_workers=self.args.n_workers, max_queue_size=self.args.batch_size*2)
            print(f'active_threads Train {threading.active_count()}')
        if 'val' in self.datatype:
            if not 'train' in self.datatype:
                self.mean_train = 0.0
                self.std_train = 1.0
                self.read_val_data()
            
            size = [np.prod(x.shape) * x.itemsize for x in self.val]
            print('VAL: {:.2f} MB loaded in memory'.format(np.sum(size) / 1e6))

            self.n_channels = self.val[0].shape[-1]

            self.single_gen_val = []
            self.single_gen_rand_val = []
            # val_dset_ = -1 # TODO implement several datasets for val dset
            tbar = tqdm.trange(len(self.val), disable=False)
            for val_dset_ in tbar:
                self.single_gen_val.append(
                    PatchExtractor(dataset_low=self.val[val_dset_], dataset_high=self.val_h[val_dset_],
                                                             label=self.labels_val[val_dset_],
                                                             patch_l=self.patch_l_eval, n_workers=self.n_workers, max_queue_size=10,
                                                             is_random=False,
                                                             border=4,
                                                             scale=self.scale,
                                                             lims_with_labels=self.lims_labels_val[val_dset_],
                                                             patches_with_labels=self.args.patches_with_labels,
                                                             two_ds=self.two_ds, use_location=args.is_use_location,is_use_queues=False,
                                                             ds_info=self.val_info[val_dset_]))
                tbar.set_description(f'active_threads {threading.active_count()}')
                if self.single_gen_val[-1].nr_patches*10 >= args.val_patches:

                    self.single_gen_rand_val.append(
                        PatchExtractor(dataset_low=self.val[val_dset_], dataset_high=self.val_h[val_dset_],
                                                             label=self.labels_val[val_dset_],
                                                             patch_l=self.patch_l_eval, n_workers=self.n_workers, max_queue_size=10,
                                                             is_random=True, max_N=args.val_patches,
                                                             border=4,
                                                             scale=self.scale, 
                                                             lims_with_labels=self.lims_labels_val[val_dset_],
                                                             patches_with_labels=self.args.patches_with_labels,
                                                             two_ds=self.two_ds, use_location=args.is_use_location,is_use_queues=False,
                                                             ds_info=self.val_info[val_dset_]))
                else:
                    print(f' Complete val dataset has {self.single_gen_val[-1].nr_patches} and it is used as random sub-sample too...')

                    self.single_gen_rand_val.append(self.single_gen_val[-1])

            #self.patch_gen_val_rand = PatchWraper(self.single_gen_rand_val, n_workers=self.args.n_workers, max_queue_size=self.args.batch_size*2)
            self.patch_gen_val_complete = PatchWraper(self.single_gen_val, n_workers=self.args.n_workers, max_queue_size=self.args.batch_size*2, name='Patch_val', is_random=False)
            print(f'active_threads Train {threading.active_count()}')
        if self.datatype == 'test':
            self.prepare_test_data()

    @conditional_decorator(no_verbose,not IS_VERBOSE_DATALOAD)
    def read_data_pairs(self, path_dict, mask_out_dict=None, is_load_lab=True, is_load_input=True,
                        is_skip_if_masked=True):

        if is_load_input:

            train = read_and_upsample_sen2(data_file=path_dict['lr'], args=self.args, roi_lon_lat=path_dict['roi'], mask_out_dict=mask_out_dict,is_skip_if_masked=is_skip_if_masked)

        else:
            train,train_h = None, None
        if is_load_lab:
            labels, lim_labels = geotif.read_labels(self.args, shp_file=path_dict['gt'], roi=path_dict['roi'], roi_with_labels=path_dict['roi_lb'],
                                           is_HR=self.is_HR_labels, ref_lr=path_dict['lr'], ref_hr=path_dict['hr'])
            if labels is not None:
                if 'age' in self.args.dataset:
                    print(f' Ages: {np.unique(labels)}')
                else:
                    labels_masked = labels.copy()
                    labels_masked[labels == -1] = np.nan
                    print(f' Densities percentiles 10,20,50,70,90 \n {np.nanpercentile(labels_masked, q=[0.1,0.2,0.5,0.7,0.9])}')
                    if 'palmcoco' in self.args.dataset:
                        if 'labels/palm' in path_dict['gt'] or 'palm_annotations' in path_dict['gt']:
                            labels = np.concatenate((labels,np.zeros_like(labels)), axis=-1)

                            print('Palm Object - class 1')
                        elif 'labels/coco/' in path_dict['gt'] or 'coco_annotations' in path_dict['gt']:
                            labels = np.concatenate((np.zeros_like(labels),labels),axis=-1)
                            print('Coco Object - class 2')
                        elif 'labels/coconutSHP/' in path_dict['gt']:
                            labels = np.concatenate((labels == 6, labels == 2), axis=-1) ## Palm is code 6 and coco code 2
                            labels = np.int32(labels) * 99 # without density Gt we just define it as 0.8 trees/pixel if there is a tree
                            print('Coco and Palm Object')
                        else:
                            raise ValueError('Label type cannot be inferred from path:\n'+path_dict['gt'])

        else:
            labels,lim_labels = None, None

        return train, None, labels, lim_labels
    
    # @conditional_decorator(no_verbose,not IS_VERBOSE_DATALOAD)
    def read_train_data(self):

        print('\n [*] Loading TRAIN data \n')

        list_id_lab = [get_id_lab(x) for x in self.args.tr]
        dict_id_lab = {x:[False,list_id_lab.index(x),None,None] for x in set(list_id_lab)}

        reload_ = True
        file_dataset =  os.path.join(self.args.model_dir.split(self.args.dataset)[0],self.args.dataset,'tmp_data')

        if os.path.isfile(file_dataset+'/dict_id_lab.json') and is_save_or_load_file:
            with open(file_dataset+'/dict_id_lab.json', 'r') as fp:
                list_keys = json.load(fp)

            list_keys_ = {x.split('barry_palm',1)[-1] for x in list_keys}
            dict_id_lab_keys_ = {x.split('barry_palm',1)[-1] for x in dict_id_lab.keys()}
            if list_keys_ == dict_id_lab_keys_:
                with np.load(file_dataset+'/arrays_train.npz', allow_pickle=True) as data:
                    self.train = data['train']
                    self.train_h = data['train_h']
                    self.labels = data['labels']
                with open(file_dataset+'/lims_labels.json', 'r') as fp:
                    self.lims_labels = json.load(fp)
                with open(file_dataset+'/train_info.json', 'r') as fp:
                    self.train_info = json.load(fp)
                basepath = self.args.save_dir.split('sparse',1)[0]
                self.train_info = check_path(self.train_info,basepath)

                reload_ = False
                print(f'loaded train data from {file_dataset} n={len(self.train)}')
            else:
                print('not all keys matched, will reload',file_dataset)
        else:
            print('file not found, will reload',file_dataset)

        if reload_:
            # Load first labels
            for _,val in tqdm.tqdm(dict_id_lab.items(), desc='loading labels'):
                _,_,labels,lim_labels = self.read_data_pairs(self.args.tr[val[1]], is_load_lab=True, is_load_input=False)
                val[2] = labels
                val[3] = lim_labels

            if USE_RAY:
                ray.init()
                print(ray.nodes()[0]['Resources'])
                data_tr = [f_read.remote(i,dict_id=dict_id_lab,mask_dict=None, read_data_pairs_func=self.read_data_pairs) for i in self.args.tr]
                self.train,self.train_h,self.labels,self.lims_labels,is_valid = zip(*ray.get(data_tr))
                ray.shutdown()
            else:
                iter_tr = tqdm.tqdm(self.args.tr, desc='loading input data')
                data_tr = [f_read(i, dict_id=dict_id_lab, mask_dict=None,read_data_pairs_func=self.read_data_pairs) for i in iter_tr]
                self.train, self.train_h, self.labels, self.lims_labels, is_valid = zip(*data_tr)

            self.train = list(compress(self.train, is_valid))
            self.train_h = list(compress(self.train_h, is_valid))
            self.labels = list(compress(self.labels, is_valid))
            self.lims_labels = list(compress(self.lims_labels, is_valid))

            self.train_info = list(compress(self.args.tr, is_valid))
            if is_save_or_load_file:
                if not os.path.exists(file_dataset):
                    os.makedirs(file_dataset)

                with open(file_dataset+'/dict_id_lab.json', 'w') as fp:
                    json.dump(list(dict_id_lab.keys()), fp, indent=4)
                np.savez(file_dataset+'/arrays_train.npz',train=self.train,train_h=self.train_h, labels=self.labels, force_zip64=True)
                with open(file_dataset+'/lims_labels.json', 'w') as fp:
                    json.dump(self.lims_labels, fp,  indent=4)
                with open(file_dataset+'/train_info.json', 'w') as fp:
                    json.dump(self.train_info, fp,  indent=4)
                print(f'saved dataset in {file_dataset} n={len(self.train)}')


        if self.args.unlabeled_data is not None:
            self.unlab = read_and_upsample_sen2(data_file=self.args.unlabeled_data, args=self.args,
                                                roi_lon_lat=self.args.roi_lon_lat_unlab)
        self.n_classes = self.labels[0].shape[-1]

        # sum_train = np.expand_dims(self.train.sum(axis=(0, 1)),axis=0)
        # self.N_pixels = self.train.shape[0] * self.train.shape[1]
        x_sum = np.sum([x.sum(axis=(0, 1)) for x in self.train], axis=0)
        x2_sum = np.sum([(x**2).sum(axis=(0, 1)) for x in self.train], axis=0)
        n = np.sum([x.shape[0] * x.shape[1] for x in self.train])

        self.mean_train = x_sum / n
        self.std_train = (x2_sum / n - (x_sum/n)**2)**0.5
        # train_flat = np.concatenate([x.reshape(-1,11) for x in self.train])
        # self.mean_train = self.train[0].mean(axis=(0, 1))
        self.max_dens = np.max([x.max() for x in self.labels])
        # self.std_train = self.train[0].std(axis=(0, 1))

        if self.args.is_empty_aerial:
            self.train[(self.labels == -1)[..., 0], :] = 2000.0
        print(str(self.mean_train))
        self.nb_bands = self.train[0].shape[-1]

        scale = self.scale
        for i_ in range(len(self.train)):
            if self.train_h[i_] is not None:
                self.train_h[i_],self.train[i_],self.labels[i_] = self.correct_shapes(
                    self.train_h[i_],self.train[i_],self.labels[i_], scale,type='Train',is_hr_lab=self.is_HR_labels)
        
        if self.args.is_padding:
            a = self.patch_l - 1
            self.train = np.pad(self.train, ((a, a), (a, a), (0, 0)), mode='constant', constant_values=0.0)
            b = a * scale
            self.train_h = np.pad(self.train_h, ((b, b), (b, b), (0, 0)), mode='constant',
                                  constant_values=0.0) if self.train_h is not None else self.train_h
            c = b if self.is_HR_labels else a
            self.labels = np.pad(self.labels, ((c, c), (c, c), (0, 0)), mode='constant', constant_values=-1.0)

            print('Padded datasets with low ={}, high={} with 0.0'.format(a, b))
        if self.args.is_use_location:
            for i_ in range(len(self.train)):
                lonlat = gp.get_location_array(self.train_info[i_]['lr'], lims=self.lims_labels[i_])
                self.train[i_] = np.concatenate((self.train[i_],lonlat), axis=-1)
            self.mean_train = np.concatenate((self.mean_train,np.array([0.,0.])))
            self.std_train = np.concatenate((self.std_train,np.array([1.,1.])))

    # @conditional_decorator(no_verbose,not IS_VERBOSE_DATALOAD)
    def read_val_data(self):

        print('\n [*] Loading VALIDATION data \n')

        maskout = {'CLD':10, # if Cloud prob is higher than 10%
                   'SCL':[3,11], # if SCL is equal to cloud or cloud shadow
                   '20m':0} # if all 20m bands are 0
        list_id_lab_val = [get_id_lab(x) for x in self.args.val]
        dict_id_lab_val = {x:[False,list_id_lab_val.index(x),None,None] for x in set(list_id_lab_val)}

        is_save_or_load_file = True
        reload_ = True
        file_dataset =  os.path.join(self.args.model_dir.split(self.args.dataset)[0],self.args.dataset,'tmp_data')
        #if 'palm4748a' in self.args.dataset:
        #    file_dataset =  os.path.join(self.args.model_dir.split(self.args.dataset)[0],'palm4748a','tmp_data')
        #    print('looking for', file_dataset)
        if os.path.isfile(file_dataset+'/dict_id_lab_val.json') and is_save_or_load_file:
            with open(file_dataset+'/dict_id_lab_val.json', 'r') as fp:
                list_keys = json.load(fp)
            list_keys_ = {x.split('barry_palm',1)[-1] for x in list_keys}
            dict_id_lab_val_keys_ = {x.split('barry_palm',1)[-1] for x in dict_id_lab_val.keys()}
            if list_keys_ == dict_id_lab_val_keys_:
                with np.load(file_dataset+'/arrays_val.npz', allow_pickle=True) as data:
                    self.val = data['val']
                    self.val_h = data['val_h']
                    self.labels_val = data['labels_val']
                with open(file_dataset+'/lims_labels_val.json', 'r') as fp:
                    self.lims_labels_val = json.load(fp)
                with open(file_dataset+'/val_info.json', 'r') as fp:
                    self.val_info = json.load(fp)
                basepath = self.args.save_dir.split('sparse',1)[0]
                self.val_info = check_path(self.val_info,basepath)

                reload_ = False
                print(f'loaded val data from {file_dataset} n={len(self.val)}')
            else:
                print('not all keys matched, will reload',file_dataset)
        else:
            print('file not found, will reload',file_dataset)

        if reload_:
            for _,val in tqdm.tqdm(dict_id_lab_val.items(), desc='loading labels'):
                _,_,labels,lim_labels = self.read_data_pairs(self.args.val[val[1]], is_load_lab=True, is_load_input=False)
                val[2] = labels
                val[3] = lim_labels

            if USE_RAY:
                ray.init()
                print(ray.nodes()[0]['Resources'])
                data_val = [f_read.remote(tr_=i,dict_id=dict_id_lab_val,mask_dict=maskout,read_data_pairs_func=self.read_data_pairs) for i in self.args.val]
                self.val,self.val_h,self.labels_val,self.lims_labels_val,is_valid = zip(*ray.get(data_val))
                ray.shutdown()
            else:
                iter_val = tqdm.tqdm(self.args.val, desc='loading input data')
                data_val = [f_read(i, dict_id=dict_id_lab_val, mask_dict=maskout,read_data_pairs_func=self.read_data_pairs) for i in iter_val]
                self.val, self.val_h, self.labels_val, self.lims_labels_val, is_valid = zip(*data_val)

            self.val = list(compress(self.val, is_valid))
            self.val_h = list(compress(self.val_h, is_valid))
            self.labels_val = list(compress(self.labels_val, is_valid))
            self.lims_labels_val = list(compress(self.lims_labels_val, is_valid))

            self.val_info = list(compress(self.args.val,is_valid))
            if is_save_or_load_file:
                if not os.path.exists(file_dataset):
                    os.makedirs(file_dataset)

                with open(file_dataset+'/dict_id_lab_val.json', 'w') as fp:
                    json.dump(list(dict_id_lab_val.keys()), fp, indent=4)
                np.savez(file_dataset+'/arrays_val.npz',val=self.val,val_h=self.val_h, labels_val=self.labels_val)
                with open(file_dataset+'/lims_labels_val.json', 'w') as fp:
                    json.dump(self.lims_labels_val, fp,  indent=4)
                with open(file_dataset+'/val_info.json', 'w') as fp:
                    json.dump(self.val_info, fp,  indent=4)
                print(f'saved dataset in {file_dataset} n={len(self.val)}')


        self.val_tilenames = [x['tilename'] for x in self.val_info]

        self.n_classes = self.labels_val[0].shape[-1]

        for i_ in range(len(self.val)):

            if self.val_h[i_] is not None:
                self.val_h[i_], self.val[i_], self.labels_val[i_] = self.correct_shapes(
                    self.val_h[i_], self.val[i_], self.labels_val[i_], self.scale, type='Val', is_hr_lab=self.is_HR_labels)

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
                    for j in range(self.n_classes):
                        plt_reg(self.labels_val[i_][...,j], self.args.model_dir + f'/val_reg_label{i_}_class{j}')
                    plots.plot_rgb(self.val[i_], file=self.args.model_dir + f'/val_LR{i_}')

                    if self.val_h[i_] is not None:
                        plots.plot_rgb(self.val_h[i_], file=self.args.model_dir+f'/val_HR{i_}', reorder=False, normalize=False)
                        np.save(self.args.model_dir + f'/val_HR{i_}', self.val_h[i_])

                    np.save(self.args.model_dir + f'/val_LR{i_}', self.val[i_])
                    np.save(self.args.model_dir + f'/val_reg_label{i_}', self.labels_val[i_])
        if self.args.is_use_location:
            for i_ in range(len(self.val)):
                lonlat = gp.get_location_array(self.val_info[i_]['lr'], lims=self.lims_labels_val[i_])
                self.val[i_] = np.concatenate((self.val[i_],lonlat), axis=-1)

    def read_test_data(self):

        print('\n [*] Loading TEST data \n')

        self.is_HR_labels = False

        self.test_h = []
        self.test = []
        self.labels_test = []
        self.lims_labels_test = []
        is_valid = []
        maskout = {'CLD':10, # if Cloud prob is higher than 10%
                   'SCL':[3,11,6], # if SCL is equal to cloud shadow, snow or water
                   '20m':0} # if all 20m bands are 0

        for test_ in self.args.test:

            test, test_h, labels_test, lim_labels_test = self.read_data_pairs(test_, mask_out_dict=maskout,
                                                                              is_skip_if_masked=False)
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
        if self.args.is_use_location:
            for i_ in range(len(self.test)):
                lonlat = gp.get_location_array(self.test_info[i_]['lr'], lims=self.lims_labels_test[i_])
                self.test[i_] = np.concatenate((self.test[i_],lonlat), axis=-1)

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
                               patch_l=self.patch_l_eval, n_workers=self.n_workers, max_queue_size=10,
                               is_random=False, border=self.args.border,
                               scale=self.scale, lims_with_labels=self.lims_labels_test[test_dset_],
                               patches_with_labels=self.args.patches_with_labels,
                               two_ds=self.two_ds, is_use_queues=False))

    def get_input_test(self, is_restart=False, as_list=False):
        if not as_list:
            if is_restart:
                self.patch_gen_test.redefine_queues()

            # return partial(self.input_fn, type='test')
            return self.input_fn(type='test')
        else:
            list_fn = []
            for id_, d_ in enumerate(self.single_gen_test):
                if is_restart:
                    d_.redefine_queues()
                # list_fn.append(partial(self.input_fn,type=f'test_complete-{id_}'))
                list_fn.append(self.input_fn(type=f'test_complete-{id_}'))
            return list_fn

    def get_input_val(self, is_restart=False, as_list=False):
        if not as_list:
            if is_restart:
                self.patch_gen_val_complete.redefine_queues()

            # return partial(self.input_fn, type='val_complete')
            return self.input_fn(type='val_complete')
        else:
            list_fn = []
            for id_, d_ in enumerate(self.single_gen_val):
                if is_restart:
                    d_.redefine_queues()
                # list_fn.append(partial(self.input_fn,type=f'val_complete-{id_}'))
                list_fn.append(self.input_fn(type=f'val_complete-{id_}'))
            return list_fn

    def normalize(self, sample):
        sample_out = {}
        for key, val in sample.items():
            
            if 'feat' in key:
                sample_out[key] = (val -  self.mean_train) /  self.std_train
            else: 
                sample_out[key] = val

        return sample_out
    # def normalize(self, features, labels):

    #     if isinstance(features,dict):
    #         features['feat_l'] -= self.mean_train
    #         features['feat_l'] /= self.std_train
    #         if self.two_ds:
    #             features['feat_lU'] -= self.mean_train
    #             features['feat_lU'] /= self.std_train
    #     else:
    #         features -= self.mean_train
    #         features /= self.std_train
    #     return features, labels
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

    def input_fn(self, type='train'):
        # np.random.seed(99)
        batch = self.batch_eval
        if type == 'train':
            ref_patch_gen = self.patch_gen
            batch = self.batch_tr
        elif type == 'val':
            ref_patch_gen = self.patch_gen_val_rand
        elif type == 'val_complete':
            ref_patch_gen = self.patch_gen_val_complete
        elif 'val_complete' in type:
            type,id_ = type.split('-')
            id_ = int(id_)
            ref_patch_gen = self.single_gen_val[id_]
        elif type == 'test':
            # ref_patch_gen = self.patch_gen_test
            raise NotImplementedError
        elif 'test_complete' in type:
            type,id_ = type.split('-')
            id_ = int(id_)
            ref_patch_gen = self.single_gen_test[id_]
        else:
            sys.exit(1)

        dict_sample = ref_patch_gen.__getitem__(0)
        gen_func = ref_patch_gen.get_iter_patch
        
        ds = tf.data.Dataset.from_generator(
                gen_func,
                output_types={k: tf.float32 if not isinstance(val,str) else tf.string for k,val in dict_sample.items()},
                output_shapes={k:val.shape if not isinstance(val,str) else () for k,val in dict_sample.items()}
            )

        if batch is None:
            batch = self.args.batch_size

        if type == 'train':
            # ds = ds.shuffle(buffer_size=batch * 5)
            if self.args.is_masking:
                ds = ds.map(self.add_spatial_masking)

        ds = ds.batch(batch)
        if 'test' not in type:
            ds = ds.map(self.normalize, num_parallel_calls=6)

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        return ds

    def get_input_fn(self, is_val_random=False):
        val_type = 'val' if is_val_random else 'val_complete'
        input_fn = None
        if self.is_training:
            input_fn = partial(self.input_fn, type='train')
        input_fn_val = partial(self.input_fn, type=val_type)

        return input_fn, input_fn_val
