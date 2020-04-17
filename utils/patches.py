from __future__ import division
import numpy as np
from skimage.transform import resize
import os, sys
import glob
import json
import pprint
from math import ceil
from scipy import ndimage
from utils.plots import plot_rgb, plot_labels, check_dims, plot_rgb_mask, plot_rgb_density


def sampleRandom(size, path, ratio=.2, filename = None, seed = None):

    index = np.zeros(size).astype(np.bool)
    if seed:
        np.random.seed(seed)
    idx = np.random.choice(size,int(size*ratio), replace = False)
    index[idx] = True
    if not filename:
        filename = os.path.join(path, '{}_val_index.npy'.format(size))
    np.save(filename, index)


def interpPatches(image_20lr, ref_shape, squeeze = False):

    image_20lr = check_dims(image_20lr)
    N, w,h, ch = image_20lr.shape

    image_20lr = np.rollaxis(image_20lr,-1,1)

    data20_interp = np.zeros(((N, ch) + ref_shape)).astype(np.float32)
    for k in range(N):
        for w in range(ch):
            data20_interp[k, w] = resize(image_20lr[k, w] / 65000, ref_shape, mode='reflect') * 65000  # bicubic

    data20_interp = np.rollaxis(data20_interp,1,4)
    if squeeze:
        data20_interp = np.squeeze(data20_interp, axis= 0)

    return data20_interp

def plot_patches(image10_, label_, clouds20_, save_png, idx, ref_shape, points = None):

    plot_rgb(image10_[...,0:3], file='{}{}_train'.format(save_png, idx))

    clouds_10 = interpPatches(clouds20_, ref_shape=ref_shape)
    Encoder = EncodeLabels()
    encoded_labels = Encoder.encode(label_, clouds_10.squeeze())
    plot_labels(encoded_labels, file='{}{}_label'.format(save_png, idx), colors=Encoder.color_map)
    plot_rgb(points, file='{}{}_points'.format(save_png, idx), percentiles=(0,100))

def extract_patch(data, upper_left_x, upper_left_y, patch, ratio = 1):
    crop_point = [upper_left_x,
                     upper_left_y,
                     upper_left_x + patch,
                     upper_left_y + patch]

    crop_point = [p * ratio for p in crop_point]
    # crop_point_hr = [p * ratio for p in crop_point_lr]
    if data is not None:
        data_cropped = data[crop_point[0]:crop_point[2], crop_point[1]:crop_point[3]]
        assert data_cropped.shape[0:2] == (patch*ratio,patch*ratio), "patch is smaller than it should be, data might be smaller than patch size"
    else:
        data_cropped = None
    return data_cropped


# input in pixel (not in tile numbers)
def crop_patch(data, upper_left_x_pix, upper_left_y_pix, patch):
    crop_point = [upper_left_x_pix,
                     upper_left_y_pix,
                     upper_left_x_pix + patch,
                     upper_left_y_pix + patch]

    # crop from image data: rows: y values, cols: x values
    data_cropped = data[crop_point[1]:crop_point[3], crop_point[0]:crop_point[2], :]

    return data_cropped



def save_random_patches(label, d10, d20, dCLD20, dPoints10 = None, file ='', filename='data', NR_CROP=8000, is_save_png=False, patch=256, val_index = None):

    if d10.shape[0:2] == d20.shape[0:2]:
        ratio = 1
    else:
        ratio = 2
    hr_shape = d10.shape[0:2]
    patch_lr, residual = divmod(patch, ratio)

    if residual != 0:
        print('{} patch size is not mutiple of 2, will reduce it to {}'.format(patch, patch_lr*ratio))
        patch = patch_lr * ratio

    n_x = d20.shape[0] - patch_lr
    n_y = d20.shape[1] - patch_lr

    if n_x *n_y < NR_CROP:
        NR_CROP = n_x * n_y

    BANDS10 = d10.shape[2]
    BANDS20 = d20.shape[2]



    if is_save_png:
        save_png = os.path.join(file, 'png_'+filename, '')
        if not os.path.exists(save_png):
            os.makedirs(save_png)
    else:
        save_png = None

    ind = np.random.choice(n_x*n_y,int(NR_CROP), replace=False)

    def get_patches(data, ratio, all_blank_index=True):
        if len(data.shape) == 3:
            data_output = np.zeros([NR_CROP, patch_lr * ratio, patch_lr * ratio, data.shape[-1]], dtype=np.float32)
        else:
            data_output = np.zeros([NR_CROP, patch_lr * ratio, patch_lr * ratio], dtype=np.float32)


        for idx, value in enumerate(ind):
            upper_left_x, upper_left_y = divmod(value, n_y)
            if not all_blank_index:
                data_temp = extract_patch(data, upper_left_x, upper_left_y, patch=patch_lr, ratio=ratio)

                # Ignore patches that have No data
                if np.unique(data_temp).size > 1 and not np.all(np.isnan(data_temp)):
                    data_output[idx] = data_temp
                else:
                    blank_index[idx] = True
            elif not blank_index[idx]:
                data_output[idx] = extract_patch(data, upper_left_x, upper_left_y, patch=patch_lr, ratio=ratio)

        if np.any(blank_index):  # if there are blank patches, remove them
            data_output = data_output[~blank_index]
        return data_output

    blank_index = np.zeros_like(ind, dtype=bool)

    image_10 = get_patches(d10, ratio=ratio, all_blank_index=False)
    del d10
    if np.all(blank_index):
        print('No Patches with data were found')
    else:
        label_10 = get_patches(label, ratio=ratio)
        del label
        points_10 = get_patches(dPoints10, ratio=ratio)
        del dPoints10

        image_20 = get_patches(d20, ratio=1)
        image_20_hr = clouds_20_hr = None
        if ratio == 2:
            d20_hr = interpPatches(d20, hr_shape, squeeze=True)
            del d20
            image_20_hr = get_patches(d20_hr, ratio=ratio)
            del d20_hr
        clouds_20 = get_patches(dCLD20, ratio=1)
        if ratio == 2:
            dCLD20_hr = interpPatches(dCLD20, hr_shape, squeeze=True)
            del dCLD20
            clouds_20_hr = get_patches(dCLD20_hr.squeeze(axis=2), ratio=ratio)
            del dCLD20_hr


        np.savez(file+filename, label10=label_10, points10=points_10,
                 data10=image_10, data20_hr = image_20_hr, cld20_hr = clouds_20_hr,
                 data20=image_20,  cld20=clouds_20)

        print('Patch size: 10m = {}, 20m = {}'.format(patch, patch_lr))
        print('Nr. band: 10m = {}, 20m = {}'.format(BANDS10, BANDS20))
        print('n_x={}, n_y={}, Max.Patches = {:.2f}K'.format(n_x, n_x, n_x * n_y / 1e3))
        print('Nr. patches = {}'.format(np.sum(~blank_index)))
        print(' [*] {}.npz saved!'.format(filename))
        if is_save_png:
            ind_1 = np.random.choice(np.arange(0,np.sum(~blank_index)),50, replace=False)
            for i, val in enumerate(ind_1):
                plot_patches(image_10[val], label_10[val].copy(), clouds_20_hr[val], save_png, i,
                             ref_shape=(patch, patch), points=points_10[val].copy())



def save_test_patches(d10, d20, file, filename = 'data', patch=128, border=4, interp=True):

    if d10.shape[0:2] == d20.shape[0:2]:
        ratio = 1
    else:
        ratio = 2
    patch20 = int(patch / ratio)


    PATCH_SIZE_HR = (patch, patch)
    PATCH_SIZE_LR = [p//ratio for p in PATCH_SIZE_HR]
    BORDER_HR = border
    BORDER_LR = BORDER_HR//ratio

    # Mirror the data at the borders to have the same dimensions as the input
    d10 = np.pad(d10, ((BORDER_HR, BORDER_HR), (BORDER_HR, BORDER_HR), (0, 0)), mode='symmetric')
    d20 = np.pad(d20, ((BORDER_LR, BORDER_LR), (BORDER_LR, BORDER_LR), (0, 0)), mode='symmetric')

    BANDS10 = d10.shape[2]
    BANDS20 = d20.shape[2]
    patchesAlongi = (d20.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    patchesAlongj = (d20.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)

    # image_20 = np.zeros((nr_patches, BANDS20) + tuple(PATCH_SIZE_LR)).astype(np.float32)
    # image_10 = np.zeros((nr_patches, BANDS10) + PATCH_SIZE_HR).astype(np.float32)


    image_10 = np.zeros([nr_patches, patch, patch, BANDS10], dtype=np.float32)
    image_20 = np.zeros([nr_patches, patch20, patch20, BANDS20], dtype=np.float32)


    print(image_20.shape)
    print(image_10.shape)

    range_i = np.arange(0, (d20.shape[0] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)) * (
        PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    range_j = np.arange(0, (d20.shape[1] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)) * (
        PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    if not (np.mod(d20.shape[0] - 2 * BORDER_LR, PATCH_SIZE_LR[0] - 2 * BORDER_LR) == 0):
        range_i = np.append(range_i, (d20.shape[0] - PATCH_SIZE_LR[0]))
    if not (np.mod(d20.shape[1] - 2 * BORDER_LR, PATCH_SIZE_LR[1] - 2 * BORDER_LR) == 0):
        range_j = np.append(range_j, (d20.shape[1] - PATCH_SIZE_LR[1]))

    print(range_i)
    print(range_j)

    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            crop_point_lr = [upper_left_i,
                             upper_left_j,
                             upper_left_i + PATCH_SIZE_LR[0],
                             upper_left_j + PATCH_SIZE_LR[1]]
            crop_point_hr = [p*ratio for p in crop_point_lr]
            image_20[pCount] = d20[crop_point_lr[0]:crop_point_lr[2],crop_point_lr[1]:crop_point_lr[3]]
            image_10[pCount] = d10[crop_point_hr[0]:crop_point_hr[2],crop_point_hr[1]:crop_point_hr[3]]

            pCount += 1

    np.savez(file + filename, data10=image_10[0:pCount], data20=image_20[0:pCount])
    print('Patch size: 10m = {}, 20m = {}'.format(patch, patch20))
    print('Nr. band: 10m = {}, 20m = {}'.format(BANDS10, BANDS20))
    print('Nr. patches = {}'.format(pCount))
    print(' [*] {}.npz saved!'.format(filename))


def recompose_images(a, border=4, size=None, show_image=False, verbose=False):
    '''
    a is an array of shape NxPxPxC
    where N is the number of patches, P is the patch size
    '''
    if len(a.shape) == 3:
        a = np.expand_dims(a,axis=-1)

    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2]-border*2
        if verbose:
            print('Patch w/o borders has dimension {}'.format(patch_size))
            print('Prediction has shape {}'.format(a.shape))
        x_tiles = int(ceil(size[0]/float(patch_size)))
        y_tiles = int(ceil(size[1]/float(patch_size)))
        if verbose: print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        if verbose: print('Image size is: {}'.format(size))
        images = np.zeros(shape=(size[1], size[0])+tuple(a.shape[3:]),
                          dtype=np.float32)

        if verbose: print(images.shape)
        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[1] - patch_size:
                ypoint = size[1] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[0] - patch_size:
                    xpoint = size[0] - patch_size
                images[ypoint:ypoint+patch_size, xpoint:xpoint+patch_size,:] = a[current_patch, border:a.shape[2]-border, border:a.shape[2]-border,:]
                current_patch += 1


    return images



def save_parameters(params, out_dir, sysargv=None, name='FLAGS'):

    params = params.__dict__

    with open(os.path.join(out_dir, name + '.txt'), 'w') as f:
        sorted_names = sorted(params.keys(), key=lambda x: x.lower())
        for key in sorted_names:
            value = params[key]
            f.write('%s:%s\n' % (key, value))

        if sysargv:
            f.write("\n\n")
            for i in sysargv:
                f.write("{} ".format(i))

    pprint.pprint(params)

    print(' '.join(sysargv))

def check_if_existing(args, folder):
    if args.data_type == 'raw':
        dataset_type = args.data_type
    else:
        dataset_type = args.data_type + str(args.patch_size)

    save_prefix = os.path.join(args.save_prefix,dataset_type)

    if args.roi_lon_lat:
        save_prefix = os.path.join(save_prefix,args.roi_lon_lat.replace(',','-'))

    if args.data_type == 'raw':
        filename_npz = os.path.join(save_prefix, folder)
        fileList = [x for x in sorted(glob.glob(filename_npz + "*.npz"))]

    elif args.data_type == 'train':
        filename_npz = os.path.join(save_prefix, folder, '')
        fileList = [x for x in sorted(glob.glob(filename_npz + "bicubic.npz"))]

    if fileList:
        print('Data already exists.. skipping it...')
        sys.exit(0)
def rename_dset_path(args):

    dataset_type = args.data_type

    return os.path.join(args.save_prefix, dataset_type)
    # return save_prefix
def save_numpy(data, args, folder, view, filename, is_save_png=True, points = None):

    save_prefix = rename_dset_path(args)
    data = data.astype(np.float32)


    if args.roi_lon_lat:
        roi_lon_lat = args.roi_lon_lat.replace(',','-')
        save_prefix = os.path.join(save_prefix,roi_lon_lat)
    else:
        roi_lon_lat = ''

    data = data.astype(np.uint8)

    if args.data_type == 'raw' or args.data_type == 'raw':
        if not os.path.exists(os.path.join(save_prefix,'rgb')):
            os.makedirs(os.path.join(save_prefix,'rgb'))

        filename_npz = os.path.join(save_prefix, folder + '_' + filename + '_' + view)

        np.savez(filename_npz, data=data)

        filename_png = os.path.join(save_prefix, 'rgb', folder + '_' + filename + '_' + view + '_')

    elif args.data_type == 'train' or args.data_type == 'validation':
        out_per_image = os.path.join(save_prefix, folder, '')

        if not os.path.isdir(out_per_image):
            os.makedirs(out_per_image)
        print('Writing files for {} dataset to:{}'.format(args.data_type,out_per_image))


        np.save(out_per_image + 'data_complete', data)
        print('Complete dataset saved as {} (No Patches)'.format(out_per_image+'data_complete'))
        filename_png =os.path.join(save_prefix, folder, '')

    else:
        print(' [!] Dataset {} not recognized '.format(args.data_type))
        sys.exit(1)

    save_parameters(args, save_prefix, sysargv=sys.argv)
    plot_rgb(data, filename_png + 'RGB')
