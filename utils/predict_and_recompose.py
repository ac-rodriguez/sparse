import os
from itertools import compress
import time
import numpy as np
from tqdm import tqdm

import utils.plots as plots
import utils.patches as patches
import utils.gdal_processing as gp
from utils.tools_tf import copy_last_ckpt, InputNorm
import tensorflow as tf

def save_m(name, m):
    with open(name, 'w') as f:
        sorted_names = sorted(m.keys(), key=lambda x: x.lower())
        for key in sorted_names:
            value = m[key]
            f.write('%s:%s\n' % (key, value))

def predict_and_recompose(trainer, reader, input_fn, patch_generator, is_hr_pred, batch_size, type_,
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None, epoch_=0):
    model_dir = trainer.model_dir
    save_dir = os.path.join(model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)
    # f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    f1 = lambda x:x
    plt_reg = lambda x: plots.plot_heatmap(f1(x), cmap='viridis', percentiles=(0, 100))  # min=-1, max=2.0,

    if not isinstance(patch_generator,list):
        patch_generator = [patch_generator]
    predictions = {'reg':[],'sem':[]}
    for id_ in  range(len(patch_generator)):

        patch = patch_generator[id_].patch_l
        border = patch_generator[id_].border

        if 'val' in type_:
            ref_data = reader.val[id_]
        elif 'train' in type_:
            ref_data = reader.train[id_]
        else:
            ref_data = reader.test[id_]

        ref_size = (ref_data.shape[1], ref_data.shape[0])
        nr_patches = patch_generator[id_].nr_patches

        batch_idxs = int(np.ceil(nr_patches / batch_size))

        if is_reg:
            pred_r_rec = np.zeros(shape=([nr_patches, patch, patch, reader.n_classes]))
        if is_sem:
            pred_c_rec = np.zeros(shape=([nr_patches, patch, patch]))
        patch_generator[id_].define_queues()
        input_iter = iter(input_fn[id_])
        # preds_iter = trainer.predict(input_fn=input_fn[id_], yield_single_examples=False, checkpoint_path=chkpt_path)

        # print('Predicting {} Patches...'.format(nr_patches))
        for idx in tqdm(range(0, batch_idxs),disable=True):
            x, _ = next(input_iter)
            p_ = trainer.model(x['feat_l'], is_training=False)
            start = idx * batch_size
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
        # if type_ == 'test':
        #     del reader
        # print(ref_size)
        ## Recompose RGB
        if is_reg:
            data_r_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border)
            predictions['reg'].append(data_r_recomposed)
        else:
            data_r_recomposed = None

        if is_sem:
            data_c_recomposed = patches.recompose_images(pred_c_rec, size=ref_size, border=border)
            predictions['sem'].append(data_c_recomposed)
        else:
            data_c_recomposed = None

        if return_array:
            print('Returning only fist array of dataset list')
            return data_r_recomposed, data_c_recomposed

    if not 'train' in type_:
        if 'val' in type_:
            ref_tiles = reader.val_tilenames
            ref_data = reader.val
        else:
            ref_tiles = reader.test_tilenames
            ref_data = reader.test

        for tile in set(ref_tiles):
            index = [tile == x for x in ref_tiles]
            if is_reg:
                reg = list(compress(predictions['reg'], index))
                val_data = list(compress(ref_data,index))
                for i_, reg_ in enumerate(reg):
                    reg_[np.isnan(val_data[i_][..., -1])] = np.nan

                shapes = set([x.shape for x in reg])
                for i, s in enumerate(shapes):
                    reg_sameshape = [x for x in reg if s == x.shape]

                    reg_sameshape = np.stack(reg_sameshape, axis=-1)
                    reg_sameshape = np.nanmedian(reg_sameshape, axis=-1)
                    for j in range(reg_sameshape.shape[-1]):
                        im = plt_reg(reg_sameshape[...,j])
                        # im.save(f'{save_dir}/{type_}_reg_pred{tile}_class{j}_{i}.png')
                        with trainer.val_writer.as_default():
                            tf.summary.image(f'{type_}/reg_pred{tile}_{i}/class{j}', np.array(im)[np.newaxis], step=epoch_)
            if is_sem:
                sem = list(compress(predictions['sem'], index))
                val_data = list(compress(ref_data, index))
                for i_, sem_ in enumerate(sem):
                    sem_[np.isnan(val_data[i_][..., -1])] = np.nan
                shapes = set([x.shape for x in sem])
                for i, s in enumerate(shapes):
                    sem_sameshape = [x for x in sem if s == x.shape]

                    sem_sameshape = np.stack(sem_sameshape, axis=-1)
                    sem_sameshape = np.nanmedian(sem_sameshape, axis=-1)

                    np.save(f'{save_dir}/{type_}_sem_pred{tile}_{i}',sem_sameshape)
                    im = plots.plot_labels(sem_sameshape, return_img=True)
                    # im.save(f'{save_dir}/{type_}_sem_pred{tile}_{i}.png')

                    with trainer.val_writer.as_default():
                        tf.summary.image(f'{type_}/sem_pred{tile}_{i}', np.array(im)[np.newaxis], step=epoch_)

    else:
        raise NotImplementedError
        print('saving as individual files')
        for id_ in range(len(patch_generator)):

            if is_reg:
                data_r_recomposed = predictions['reg'][id_]

                np.save('{}/{}_reg_pred{}'.format(save_dir, type_,id_), data_r_recomposed)
                for i in range(data_r_recomposed.shape[-1]):
                    raise NotImplementedError
                    plt_reg(data_r_recomposed[...,i], '{}/{}_reg_pred_class{}_{}'.format(save_dir, type_,i,id_))
            if is_sem:
                data_c_recomposed = predictions['sem'][id_]
                np.save('{}/{}_sem_pred{}'.format(save_dir, type_,id_), data_c_recomposed)
                plots.plot_labels(data_c_recomposed, '{}/{}_sem_pred{}'.format(save_dir, type_,id_))


def predict_and_recompose_individual(trainer, reader, input_fn, patch_generator, is_hr_pred, batch_size, type_,
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None):
    model_dir = trainer.model_dir
    save_dir = os.path.join(model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)

    # f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    f1 = lambda x: x
    plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, cmap='viridis',
                                                 percentiles=(0, 100))  # min=-1, max=2.0,
    # copy chkpt
    if chkpt_path is not None:
        trainer.model.inputnorm = InputNorm(n_channels=11)
        trainer.model.load_weights(chkpt_path)
        
    if not isinstance(patch_generator, list):
        patch_generator = [patch_generator]

    if 'test' in type_:

        is_save_class_prob = True
        is_save_georef = True
    else:
        is_save_class_prob = False
        is_save_georef = False

    if is_save_class_prob:
        semsuffix = 'classprob'
    else:
        semsuffix = 'sem'

    for id_ in range(len(patch_generator)):

        patch = patch_generator[id_].patch_l
        border = patch_generator[id_].border

        if 'val' in type_:
            ref_data = reader.val[id_]
        elif 'train' in type_:
            ref_data = reader.train[id_]
        else:
            ref_data = reader.test[id_]

        if 'val' in type_:
            ref_info = reader.val_info[id_]
        else:
            ref_info = reader.test_info[id_]
        lr_filename = ref_info['lr']
        refDataset = gp.get_jp2(lr_filename, 'B03', res=10)
        data_name = [x for x in lr_filename.split('/') if 'SAFE' in x][0]
        ref_size = (ref_data.shape[1], ref_data.shape[0])
        nr_patches = patch_generator[id_].nr_patches

        batch_idxs = int(np.ceil(nr_patches / batch_size))

        if is_reg:
            pred_r_rec = np.zeros(shape=([nr_patches, patch, patch, reader.n_classes]))
        if is_sem:
            if is_save_class_prob:
                pred_c_rec = np.zeros(shape=([nr_patches, patch, patch, reader.n_classes+1]))
            else:
                pred_c_rec = np.zeros(shape=([nr_patches, patch, patch]))

        patch_generator[id_].define_queues()
        input_iter = iter(input_fn[id_])

        for idx in tqdm(range(0, batch_idxs), disable=None):
            x = next(input_iter)
            if 'test' in type_: x = trainer.model.inputnorm(x)
            p_ = trainer.model(x, is_training=False)
            start = idx * batch_size
            last_batch = nr_patches - start
            if is_reg:
                stop = start + p_['reg'].shape[0]
                if stop > nr_patches:
                    pred_r_rec[start:stop] = p_['reg'][0:last_batch]
                else:
                    pred_r_rec[start:stop] = p_['reg']
            if is_sem:
                stop = start + p_['sem'].shape[0]
                if is_save_class_prob:
                    f = lambda x: x
                else:
                    f = lambda x:np.argmax(x,axis=-1)
                if stop > nr_patches:
                    pred_c_rec[start:stop] = f(p_['sem'][0:last_batch])
                else:
                    pred_c_rec[start:stop] = f(p_['sem'])
        if type_ == 'test':
            reader.test[id_] = None
        print(ref_size)
        ## Recompose RGB
        if is_reg:
            data_r_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border)
            data_r_recomposed[np.isnan(ref_data[..., 0])] = np.nan

            if not return_array:
                fname = f'{save_dir}/{data_name}-{type_}_preds_reg'
                if is_save_georef:
                    # for opt in [0,1,2,3]:
                        # t0 = time.time()
                    gp.rasterize_numpy(data_r_recomposed, refDataset,
                                    filename=fname+'.tif', type='float32',
                                    roi_lon_lat=ref_info['roi'],options=2)
                    # print(f'total time option {opt} {time.time()-t0:.3f}')

                else:
                    np.save(fname, data_r_recomposed)
                    for i in range(data_r_recomposed.shape[-1]):
                        plt_reg(data_r_recomposed[..., i],
                                fname)
        else:
            data_r_recomposed = None

        if is_sem:
            data_c_recomposed = patches.recompose_images(pred_c_rec, size=ref_size, border=border)
            data_c_recomposed[np.isnan(ref_data[..., 0])] = np.nan
            if not return_array:
                fname = f'{save_dir}/{data_name}-{type_}_preds_{semsuffix}'
                if is_save_georef:
                    gp.rasterize_numpy(data_c_recomposed, refDataset,
                                        filename=fname+'.tif', type='float32',
                                        roi_lon_lat=ref_info['roi'], options=2)
                else:
                    np.save(fname, data_c_recomposed)
                    if not is_save_class_prob:
                        plots.plot_labels(data_c_recomposed, fname)
                    else:
                        raise NotImplementedError('plot class maps pending')
        else:
            data_c_recomposed = None
        if return_array:
            print('Returning only fist array of dataset list')
            return data_r_recomposed, data_c_recomposed

