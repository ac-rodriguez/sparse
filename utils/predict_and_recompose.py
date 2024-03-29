import os
from itertools import compress
import time
import numpy as np
from tqdm import tqdm, trange

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



def predict_and_recompose_with_metrics(trainer, reader, args, prefix, epoch_):
    #  input_fn, patch_generator, is_hr_pred, batch_size, type_,
                        #   prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None,
                        #    epoch_=0,is_dropout=False, n_eval_dropout=5, is_input_norm=False):
    model_dir = trainer.model_dir
    save_dir = os.path.join(model_dir, prefix)
    batch_size = args.batch_size_eval
    patch_generator = reader.single_gen_val
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    #if m is not None:
    #    save_m(save_dir + '/metrics.txt', m)
    plt_reg = lambda x: plots.plot_heatmap(x, cmap='viridis', percentiles=(0, 100), min=0, max=2.0)

    if not isinstance(patch_generator,list):
        patch_generator = [patch_generator]
    predictions = {'reg':[],'sem':[]}
    for id_ in  trange(len(patch_generator), disable=False ,desc='predicting each dataset'):
        patch = patch_generator[id_].patch_l
        border = patch_generator[id_].border
        #assert 'val' in type_
        ref_data = reader.val[id_]

        ref_size = (ref_data.shape[1], ref_data.shape[0])
        nr_patches = patch_generator[id_].nr_patches

        batch_idxs = int(np.ceil(nr_patches / batch_size))

        # if is_reg:
        pred_r_rec = np.zeros(shape=([nr_patches, patch, patch, reader.n_classes]))
        # if is_sem:
        pred_c_rec = np.zeros(shape=([nr_patches, patch, patch]))
        # patch_generator[id_].define_queues()
        # if input_fn is None:
        input_fn_ = reader.input_fn(type=f'val_complete-{id_}')
        input_iter = iter(input_fn_)
        # else:
            # # patch_generator[id_].redefine_queues()
            # input_iter = iter(input_fn[id_])

        for idx in trange(batch_idxs,disable=True):
            x = next(input_iter)
            if not args.is_train:
                x['feat_l'] = trainer.model.inputnorm(x['feat_l'])
            if args.is_dropout_uncertainty:
                p_ = trainer.forward_ntimes(x, is_training=False,n=args.n_eval_dropout, return_moments=False)
            else:
                p_ = trainer.model(x, is_training=False)
            start = idx * batch_size
            stop = start + p_['reg'].shape[0]
            if stop > nr_patches:
                last_batch = nr_patches - start
                pred_r_rec[start:stop] = p_['reg'][0:last_batch]
            else:
                pred_r_rec[start:stop] = p_['reg']

            stop = start + p_['sem'].shape[0]
            if stop > nr_patches:
                last_batch = nr_patches - start
                pred_c_rec[start:stop] = np.argmax(p_['sem'][0:last_batch], axis=-1)
            else:
                pred_c_rec[start:stop] = np.argmax(p_['sem'], axis=-1)
        data_r_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border)
        predictions['reg'].append(data_r_recomposed)

        data_c_recomposed = patches.recompose_images(pred_c_rec, size=ref_size, border=border)
        predictions['sem'].append(data_c_recomposed)

    ref_data = reader.val

    ref_names = [x.name for x in patch_generator]

    for tile in tqdm(set(ref_names), desc='aggregating per location'):
        index = [tile == x for x in ref_names]
        reg = list(compress(predictions['reg'], index))
        val_data = list(compress(ref_data,index))
        for i_, reg_ in enumerate(reg):
            reg_[np.isnan(val_data[i_][..., -1])] = np.nan

        assert len(set([x.shape for x in reg])) == 1
        
        reg = np.stack(reg, axis=-1)
        reg = np.nanmedian(reg, axis=-1)
        # if is_add_labels:
        val_labels = list(compress(reader.labels_val,index))[0]
        
        for j in range(reg.shape[-1]):
            with trainer.val_writer.as_default():
                im = plt_reg(reg[...,j])
                im_lab = plt_reg(val_labels[...,j])
                img_summary = np.concatenate((np.array(im),np.array(im_lab)),axis=1)[np.newaxis]
                tf.summary.image(f'val/{tile}/reg/class{j}', img_summary, step=epoch_)
                trainer.update_sum_val_aggr_reg(reg[...,j],val_labels[...,j], f'{tile}')

    metrics = trainer.summaries_val(step=epoch_)

    save_m(save_dir + '/metrics.txt', metrics)
    return metrics

def predict_and_recompose(trainer, reader, input_fn, patch_generator, is_hr_pred, batch_size, type_,
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None,
                           epoch_=0,is_dropout=False, n_eval_dropout=5, is_input_norm=False):
    model_dir = trainer.model_dir
    save_dir = os.path.join(model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)
    # f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    f1 = lambda x:x
    plt_reg = lambda x: plots.plot_heatmap(f1(x), cmap='viridis', percentiles=(0, 100), min=0, max=2.0)

    if not isinstance(patch_generator,list):
        patch_generator = [patch_generator]
    predictions = {'reg':[],'sem':[]}
    for id_ in  trange(len(patch_generator), disable=False ,desc='predicting each dataset'):
    # for id_ in  trange(10, disable=False):

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
        # patch_generator[id_].define_queues()
        if input_fn is None:
            input_fn_ = reader.input_fn(type=f'val_complete-{id_}')
            input_iter = iter(input_fn_)
        else:
            # patch_generator[id_].redefine_queues()
            input_iter = iter(input_fn[id_])

        # print('Predicting {} Patches...'.format(nr_patches))
        for idx in trange(batch_idxs,disable=True):
            x = next(input_iter)
            if is_input_norm: x['feat_l'] = trainer.model.inputnorm(x['feat_l'])
            if is_dropout:
                p_ = trainer.forward_ntimes(x, is_training=False,n=n_eval_dropout, return_moments=False)
            else:
                p_ = trainer.model(x, is_training=False)
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
        # patch_generator[id_].finish_threads()
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

    is_add_labels = True and hasattr(reader,'labels_val') 

    if not 'train' in type_:
        if 'val' in type_:
            ref_tiles = reader.val_tilenames
            ref_data = reader.val
        else:
            ref_tiles = reader.test_tilenames
            ref_data = reader.test

        for tile in tqdm(set(ref_tiles), desc='aggregating per location'):
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
                    if is_add_labels:
                        val_labels = list(compress(reader.labels_val,index))
                        val_labels = [x for x in val_labels if s == x.shape][0]
                    
                    for j in range(reg_sameshape.shape[-1]):
                        with trainer.val_writer.as_default():
                            im = plt_reg(reg_sameshape[...,j])
                            tf.summary.image(f'{type_}/{tile}_{i}/reg_pred/class{j}', np.array(im)[np.newaxis], step=epoch_)

                            if is_add_labels: 
                                im_lab = plt_reg(val_labels[...,j])
                                tf.summary.image(f'{type_}/{tile}_{i}/reg_label/class{j}', np.array(im_lab)[np.newaxis], step=epoch_)

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
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None, compression='0'):
    model_dir = trainer.model_dir
    save_dir = os.path.join(model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)

    # f1 = lambda x: (np.where(x == -1, x, x * (2.0 / reader.max_dens)) if is_hr_pred else x)
    f1 = lambda x: x
    plt_reg = lambda x, file: plots.plot_heatmap(f1(x), file=file, cmap='viridis',
                                                 percentiles=(0, 100))  # min=-1, max=2.0,
        
    if not isinstance(patch_generator, list):
        patch_generator = [patch_generator]

    assert 'test' in type_
    assert not is_sem and is_reg,'forward pass needs to change for class prob'
    semsuffix = 'classprob'

    # if 'test' in type_:
    #     is_save_class_prob = True
    #     is_save_georef = True
    # else:
    #     is_save_class_prob = False
    #     is_save_georef = False

    # if is_save_class_prob:
    #     semsuffix = 'classprob'
    # else:
    #     raise NotImplementedError
    #     semsuffix = 'sem'

    for id_ in range(len(patch_generator)):

        patch = patch_generator[id_].patch_l
        border = patch_generator[id_].border

        # if 'val' in type_:
        #     ref_data = reader.val[id_]
        # elif 'train' in type_:
        #     ref_data = reader.train[id_]
        # else:
        ref_data = reader.test[id_]
        # if 'val' in type_:
        #     ref_info = reader.val_info[id_]
        # else:
        ref_info = reader.test_info[id_]

        lr_filename = ref_info['lr']
        refDataset = gp.get_jp2(lr_filename, 'B03', res=10)
        data_name = [x for x in lr_filename.split('/') if 'SAFE' in x][0]
        ref_size = (ref_data.shape[1], ref_data.shape[0])
        nr_patches = patch_generator[id_].nr_patches

        batch_idxs = int(np.ceil(nr_patches / batch_size))
        keys_reconstruct = ['reg' ,'last']

        pred_rec = {key:[] for key in keys_reconstruct}
        patch_generator[id_].define_queues()
        if input_fn is None:
            input_fn_ = reader.input_fn(type=f'test_complete-{id_}')
            input_iter = iter(input_fn_)
        else:
            patch_generator[id_].redefine_queues()
            input_iter = iter(input_fn[id_])

        for idx in tqdm(range(0, batch_idxs), disable=None):
            x = next(input_iter)
            if 'test' in type_: x['feat_l'] = trainer.model.inputnorm(x['feat_l'])
            if isinstance(chkpt_path,list):
                p_ = trainer.forward_ensemble(x,is_training=False,ckpt_list=chkpt_path)
            else:
                p_ = trainer.model(x, is_training=False)
            start = idx * batch_size

            for key in keys_reconstruct:
                stop = start + p_[key].shape[0]
                items_slice = batch_size if stop <= nr_patches else nr_patches - start
                if key == 'last':
                    last_ = p_['last'][0:items_slice].numpy()
                    last_[np.isnan(x['feat_l'][0:items_slice,..., 0].numpy())] = np.nan
                    last_ = last_[:,border:-border,border:-border]
                    x_sum = np.nansum(last_,axis=(1,2))
                    x2_sum = np.nansum(last_**2,axis=(1,2))
                    n = np.sum(1- np.any(np.isnan(last_),axis=-1),axis=(1,2))
                    lonlat_corner = x['feat_l'][0:items_slice,border,border,11:].numpy()

                    patch_ids = np.arange(start,stop)
                    pred_rec[key].append([x_sum,x2_sum,n, lonlat_corner, patch_ids])
                else:
                    pred_rec[key].append(p_[key][0:items_slice].numpy())

        if 'test' in type_: # clean memory from input data
            reader.test[id_] = None
        print(ref_size)
        ## Recompose RGB
        data_recomposed = {}
        if 'last' in keys_reconstruct:

            fname = f'{save_dir}/{data_name}-{type_}_preds_last'

            x_sum, x2_sum, n, lonlat, patch_ids = zip(*pred_rec['last'])

            x_sum = np.concatenate(x_sum)
            x2_sum = np.concatenate(x2_sum)
            n = np.concatenate(n)
            lonlat = np.concatenate(lonlat)
            patch_ids = np.concatenate(patch_ids)

            np.savez(fname+'.npz',x_sum=x_sum,x2_sum=x2_sum,
                    n=n, lonlat=lonlat, patch_ids=patch_ids, patch_size=patch,border=border)
            print('saved',fname+'.npz')

        if 'reg' in keys_reconstruct:
            pred_rec['reg'] = np.concatenate(pred_rec['reg'], axis=0)
            data_recomposed['reg'] = patches.recompose_images(pred_rec['reg'], size=ref_size, border=border)
            data_recomposed['reg'][np.isnan(ref_data[..., 0])] = np.nan

            # data_r_recomposed = patches.recompose_images(pred_r_rec, size=ref_size, border=border)
            # data_r_recomposed[np.isnan(ref_data[..., 0])] = np.nan

            if not return_array:
                fname = f'{save_dir}/{data_name}-{type_}_preds_reg'
                # for opt in [0,1.1,1.2,1.3,2.1,2.2,2.3]:
                # for opt in [0]:
                t0 = time.time()
                data_recomposed['reg'] = data_recomposed['reg'].reshape(ref_size+(-1,))
                gp.rasterize_numpy(data_recomposed['reg'], refDataset,
                                filename=fname+'.tif', type='float32',
                                roi_lon_lat=ref_info['roi'],compression=compression)
                print(f'total time compression {compression} {time.time()-t0:.3f}')

        if 'sem' in keys_reconstruct:
            raise NotImplementedError
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

        # if return_array:
        #     print('Returning only fist array of dataset list')
        #     return data_r_recomposed, data_c_recomposed

def predict_and_recompose_individual_MC(trainer, reader, input_fn, patch_generator, is_hr_pred, batch_size, type_,
                          prefix='', is_reg=True, is_sem=True, return_array=False, m=None, chkpt_path=None, mc_repetitions=1, is_ensemble=False,
                          compression='0'):
    
    save_dir = os.path.join(trainer.model_dir, prefix)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    if m is not None:
        save_m(save_dir + '/metrics.txt', m)

        
    if not isinstance(patch_generator, list):
        patch_generator = [patch_generator]

    assert 'test' in type_
    assert not is_sem and is_reg,'forward pass needs to change for class prob'
    semsuffix = 'classprob'

    for id_ in range(len(patch_generator)):

        patch = patch_generator[id_].patch_l
        border = patch_generator[id_].border

        ref_data = reader.test[id_]
        ref_info = reader.test_info[id_]

        lr_filename = ref_info['lr']
        refDataset = gp.get_jp2(lr_filename, 'B03', res=10)
        data_name = [x for x in lr_filename.split('/') if x.endswith('.SAFE') or x.endswith('.zip')][0]
        if data_name.endswith('.zip'):
            data_name = data_name.replace('.zip','.SAFE') # Just to keep consistent naming
        ref_size = (ref_data.shape[1], ref_data.shape[0])
        nr_patches = patch_generator[id_].nr_patches

        batch_idxs = int(np.ceil(nr_patches / batch_size))
        # batch_idxs = 10
        keys_reconstruct = ['reg' ,'last']
        # (last dim for (x_sum and x2_sum))
        #if 'reg' in keys_reconstruct: 
        #    pred_rec['reg'] = np.zeros(shape=([nr_patches, patch, patch, reader.n_classes, 2]))
        pred_rec = {key:[] for key in keys_reconstruct}
        # patch_generator[id_].define_queues()
        if input_fn is None:
            input_fn_ = reader.input_fn(type=f'test_complete-{id_}')
            input_iter = iter(input_fn_)
        else:
            # patch_generator[id_].redefine_queues()
            input_iter = iter(input_fn[id_])            

        for idx in tqdm(range(0, batch_idxs), disable=None, desc='predicting'):
            x = next(input_iter)
            if 'test' in type_: x['feat_l'] = trainer.model.inputnorm(x['feat_l'])
            if is_ensemble:
                p_ = trainer.forward_ensemble(x,is_training=False,ckpt_list=chkpt_path)
            else:
                p_ = trainer.forward_ntimes(x, is_training=False, n=mc_repetitions)
            start = idx * batch_size
            
            for key in keys_reconstruct:
                stop = start + p_[key].shape[0]
                items_slice = batch_size if stop <= nr_patches else nr_patches - start
                if key == 'last':
                    last_ = p_['last'][0:items_slice].numpy()
                    last_[np.isnan(x['feat_l'][0:items_slice,..., 0].numpy())] = np.nan
                    last_ = last_[:,border:-border,border:-border]
                    x_sum = np.nansum(last_,axis=(1,2))
                    x2_sum = np.nansum(last_**2,axis=(1,2))
                    n = np.sum(1- np.any(np.isnan(last_),axis=-1),axis=(1,2))
                    lonlat_corner = x['feat_l'][0:items_slice,border,border,11:].numpy()

                    patch_ids = np.arange(start,stop)
                    pred_rec[key].append([x_sum,x2_sum,n, lonlat_corner, patch_ids])
                else:
                    pred_rec[key].append(p_[key][0:items_slice].numpy())
          
        reader.test[id_] = None
        print(ref_size)
        ## Recompose RGB
        # data_recomposed = {key:[] for key in keys_reconstruct}
        data_recomposed = {}

        if 'last' in keys_reconstruct:

            fname = f'{save_dir}/{data_name}-{type_}_preds_last'

            x_sum, x2_sum, n, lonlat, patch_ids = zip(*pred_rec['last'])

            x_sum = np.concatenate(x_sum)
            x2_sum = np.concatenate(x2_sum)
            n = np.concatenate(n)
            lonlat = np.concatenate(lonlat)
            patch_ids = np.concatenate(patch_ids)

            np.savez(fname+'.npz',x_sum=x_sum,x2_sum=x2_sum,
                    n=n, lonlat=lonlat, patch_ids=patch_ids, patch_size=patch,border=border)
            print('saved',fname+'.npz')


        if 'reg' in keys_reconstruct:
            pred_rec['reg'] = np.concatenate(pred_rec['reg'], axis=0)
            data_recomposed['reg'] = patches.recompose_images(pred_rec['reg'], size=ref_size, border=border)
            data_recomposed['reg'][np.isnan(ref_data[..., 0])] = np.nan


            if not return_array:
                fname = f'{save_dir}/{data_name}-{type_}_preds_reg'
                # for opt in [0,1.1,1.2,1.3,2.1,2.2,2.3]:
                #for opt in [1.2]:
                t0 = time.time()
                data_recomposed['reg'] = data_recomposed['reg'].reshape(ref_size+(-1,))
                gp.rasterize_numpy(data_recomposed['reg'], refDataset,
                                filename=fname+'.tif', type='float32',
                                roi_lon_lat=ref_info['roi'],compression=compression)
                print(f'total time compression {compression} {time.time()-t0:.3f}')

