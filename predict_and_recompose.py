import os
from itertools import compress

import numpy as np
from tqdm import tqdm



import plots
import patches
import gdal_processing as gp
from tools_tf import copy_last_ckpt


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

    # f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    f1 = lambda x:x
    plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, cmap='viridis',
                                                 percentiles=(0, 100))  # min=-1, max=2.0,
    # copy chkpt
    if chkpt_path is None and prefix != '':
        copy_last_ckpt(model_dir, prefix)
    if not isinstance(patch_generator,list):
        patch_generator = [patch_generator]
    predictions = {'reg':[],'sem':[]}
    for id_ in  range(len(patch_generator)):

        if is_hr_pred:
            patch = patch_generator[id_].patch_h
            border = patch_generator[id_].border_lab
            if 'val' in type_:
                ref_data = reader.val_h[id_]
            elif 'train' in type_:
                ref_data = reader.train_h[id_]
            else:
                ref_data = reader.test_h[id_]
        else:
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
        preds_iter = model.predict(input_fn=input_fn[id_], yield_single_examples=False, checkpoint_path=chkpt_path)

        print('Predicting {} Patches...'.format(nr_patches))
        for idx in tqdm(range(0, batch_idxs),disable=None):
            p_ = next(preds_iter)
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
        print(ref_size)
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
                reg = np.stack(reg, axis=-1)
                reg = np.nanmedian(reg, axis=-1)
                for i in range(reg.shape[-1]):
                    plt_reg(reg[...,i], '{}/{}_reg_pred{}_class{}'.format(save_dir, type_,tile,i))
            if is_sem:
                sem = list(compress(predictions['sem'], index))
                val_data = list(compress(ref_data, index))
                for i_, sem_ in enumerate(sem):
                    sem_[np.isnan(val_data[i_][..., -1])] = np.nan
                sem = np.stack(sem, axis=-1)
                sem = np.nanmedian(sem, axis=-1)

                np.save('{}/{}_sem_pred{}'.format(save_dir, type_, tile), sem)
                plots.plot_labels(sem, '{}/{}_sem_pred{}'.format(save_dir, type_,tile))

    else:
        print('saving as individual files')
        for id_ in range(len(patch_generator)):

            if is_reg:
                data_r_recomposed = predictions['reg'][id_]

                np.save('{}/{}_reg_pred{}'.format(save_dir, type_,id_), data_r_recomposed)
                for i in range(data_r_recomposed.shape[-1]):
                    plt_reg(data_r_recomposed[...,i], '{}/{}_reg_pred_class{}_{}'.format(save_dir, type_,i,id_))
            if is_sem:
                data_c_recomposed = predictions['sem'][id_]
                np.save('{}/{}_sem_pred{}'.format(save_dir, type_,id_), data_c_recomposed)
                plots.plot_labels(data_c_recomposed, '{}/{}_sem_pred{}'.format(save_dir, type_,id_))


def predict_and_recompose_individual(model, reader, input_fn, patch_generator, is_hr_pred, batch_size, type_,
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None):
    model_dir = model.model_dir
    save_dir = os.path.join(model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)

    # f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    f1 = lambda x: x
    plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, cmap='viridis',
                                                 percentiles=(0, 100))  # min=-1, max=2.0,
    # copy chkpt
    if chkpt_path is None and prefix != '':
        copy_last_ckpt(model_dir, prefix)
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

        if is_hr_pred:
            patch = patch_generator[id_].patch_h
            border = patch_generator[id_].border_lab
            if 'val' in type_:
                ref_data = reader.val_h[id_]
            elif 'train' in type_:
                ref_data = reader.train_h[id_]
            else:
                ref_data = reader.test_h[id_]
        else:
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

        hook = None
        patch_generator[id_].define_queues()
        preds_iter = model.predict(input_fn=input_fn[id_], yield_single_examples=False, hooks=hook,
                                   checkpoint_path=chkpt_path)


        print('Predicting {} Patches...'.format(nr_patches))
        for idx in tqdm(range(0, batch_idxs), disable=None):
            p_ = next(preds_iter)
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
                    gp.rasterize_numpy(data_r_recomposed, refDataset,
                                       filename=fname+'.tif', type='float32',
                                       roi_lon_lat=ref_info['roi'])
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
                                       roi_lon_lat=ref_info['roi'])
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