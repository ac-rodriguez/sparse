import os
import gdal
import argparse
import numpy as np
import glob

import gdal_processing as gp


parser = argparse.ArgumentParser(description="median per tile calculation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir", type=str, default="",
                    help="Path to the directory containing the predicted SAFE files")
parser.add_argument("--save_dir", default=None)
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="overwrite predictions already in folder")
args = parser.parse_args()


if args.save_dir is None:
    args.save_dir = os.path.dirname(args.data_dir)

reg_dirs = glob.glob(args.data_dir+'/*preds_reg.tif')
tilename = reg_dirs[0].split('/')[-1].split('_')[4:6]
tilename = '_'.join(tilename)


reg_filename = f'{args.save_dir}/{tilename}_preds_reg.tif'
if not os.path.isfile(reg_filename) or args.is_overwrite:
    arrays = []
    for file in reg_dirs:
        print('reading', file)
        ds = gdal.Open(file)
        nbands = ds.RasterCount
        data_ = [ds.GetRasterBand(x+1).ReadAsArray() for x in range(nbands)]
        data_ = np.stack(data_, axis=-1)
        arrays.append(data_)

    arrays = np.stack(arrays, axis=-1)
    print('computing median value')
    arrays = np.nanmedian(arrays, axis=-1)

    gp.rasterize_numpy(arrays,reg_dirs[0],filename=reg_filename,type='float32')


classprob_filename = f'{args.save_dir}/{tilename}_preds_classprob.tif'
sem_filename = f'{args.save_dir}/{tilename}_preds_sem.tif'

if not os.path.isfile(sem_filename) or not os.path.isfile(classprob_filename) or args.is_overwrite :
    sem_dirs = glob.glob(args.data_dir + '/*preds_classprob.tif')

    arrays = []
    for file in sem_dirs:
        print('reading', file)
        ds = gdal.Open(file)
        nbands = ds.RasterCount
        data_ = [ds.GetRasterBand(x + 1).ReadAsArray() for x in range(nbands)]
        data_ = np.stack(data_, axis=-1)
        arrays.append(data_)

    arrays = np.stack(arrays, axis=-1)
    print('computing median value')
    arrays = np.nanmedian(arrays, axis=-1)

    if not os.path.isfile(classprob_filename) or args.is_overwrite:

        gp.rasterize_numpy(arrays, sem_dirs[0], filename=classprob_filename, type='float32')

    if not os.path.isfile(sem_filename) or args.is_overwrite:

        # nanargmax does not handle properly pixels with all Nan values
        mask = np.isnan(arrays)
        mask = np.all(mask,axis=-1)

        arrays[mask] = -np.inf

        classes = np.nanargmax(arrays, axis=-1)
        classes[mask] = 99
        gp.rasterize_numpy(classes, sem_dirs[0], filename=sem_filename)

