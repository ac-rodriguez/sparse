import os
import gdal
import argparse
import socket
import numpy as np
import glob
from scipy import stats
import gdal_processing as gp


parser = argparse.ArgumentParser(description="median per tile calculation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir", type=str, default="",
                    help="Path to the directory containing the predicted SAFE files")
parser.add_argument("--save_dir", default=None)
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="overwrite predictions already in folder")
parser.add_argument("--is-clip-to-countries", default=False, action="store_true",
                    help="clip values outside landdefined areas")
parser.add_argument("--is-reg-only", default=False, action="store_true",
                    help=" compute regression output only (omit semantic mask)")
parser.add_argument("--is-avgprobs", default=False, action="store_true",
                    help="if true: average probabilities and then compute argmax, else mayority voting of classes")
parser.add_argument("--nan-class", default=99, type=int,
                    help="Value of output pixels with nan values in all the input datasets")
args = parser.parse_args()


def nanargmax(data, nanclass=args.nan_class):
    # nanargmax does not handle properly pixels with all Nan values
    mask = np.isnan(data)
    mask = np.all(mask, axis=-1)

    data[mask] = -np.inf

    classes = np.nanargmax(data, axis=-1)
    classes[mask] = nanclass
    return classes


def clipcountries(arrays, ref_data):
    if 'pf-pc' in socket.gethostname():
        PATH = '/scratch/andresro/leon_igp'
    else:
        PATH = '/cluster/work/igp_psr/andresro'
    shpfile = PATH + '/barry_palm/data/labels/countries/3countries.shp'
    shp_raster = gp.rasterize_polygons(shpfile, ref_data, as_bool=True)
    arrays[~shp_raster] = args.nan_class

    return arrays


if args.save_dir is None:
    args.save_dir = os.path.dirname(args.data_dir)
tilename = os.path.basename(args.data_dir)
if tilename.startswith('T'):
    # Add all the orbits in the tile
    path = os.path.dirname(args.data_dir)
    reg_dirs = glob.glob(f'{path}/*{tilename}/*preds_reg.tif')
else:
    reg_dirs = glob.glob(args.data_dir+'/*preds_reg.tif')



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
    arrays[np.isnan(arrays)] = args.nan_class

    if args.is_clip_to_countries:
        arrays = clipcountries(arrays,ref_data=reg_dirs[0])
    gp.rasterize_numpy(arrays,reg_dirs[0],filename=reg_filename,type='float32')


classprob_filename = f'{args.save_dir}/{tilename}_preds_classprob.tif'
is_compute_class = not os.path.isfile(classprob_filename) or args.is_overwrite

sem_filename = f'{args.save_dir}/{tilename}_preds_sem.tif'

if args.is_avgprobs:
    sem_filename = sem_filename.replace('sem.tif','semA.tif')
    is_compute_class = False

is_compute_sem = (not os.path.isfile(sem_filename) or args.is_overwrite) and not args.is_reg_only

if is_compute_class or is_compute_sem:
    if tilename.startswith('T'):
        # Add all the orbits in the tile
        path = os.path.dirname(args.data_dir)
        sem_dirs = glob.glob(f'{path}/*{tilename}/*preds_classprob.tif')
    else:
        sem_dirs = glob.glob(args.data_dir + '/*preds_classprob.tif')

    arrays = []
    for file in sem_dirs:
        print('reading', file)
        ds = gdal.Open(file)
        nbands = ds.RasterCount
        data_ = [ds.GetRasterBand(x + 1).ReadAsArray() for x in range(nbands)]
        data_ = np.stack(data_, axis=-1)
        arrays.append(data_)

    if args.is_avgprobs:
        arrays = np.stack(arrays, axis=-1)
        print('computing median value of probs and then argmax')
        arrays = np.nanmedian(arrays, axis=-1)

        if is_compute_class:
            if args.is_clip_to_countries:
                arrays = clipcountries(arrays, ref_data=sem_dirs[0])
            gp.rasterize_numpy(arrays, sem_dirs[0], filename=classprob_filename, type='float32')

        if is_compute_sem:
            arrays = nanargmax(arrays)
            if args.is_clip_to_countries:
                arrays = clipcountries(arrays, ref_data=reg_dirs[0])
            gp.rasterize_numpy(arrays, sem_dirs[0], filename=sem_filename)
    else:
        print('computing argmax and then majority voting')
        arrays = [nanargmax(x) for x in arrays]
        arrays = np.stack(arrays,axis=-1)
        mask = arrays == args.nan_class
        # arrays = np.float16(arrays)
        # arrays[mask] = np.nan
        # arrays = np.nanpercentile(arrays, 50, axis=-1, interpolation='nearest')
        arrays = np.ma.masked_array(data=arrays,mask=mask)
        arrays, _ = stats.mstats.mode(arrays,axis=-1)
        arrays[np.all(mask,axis=-1)] = args.nan_class
        # arrays = np.int32(arrays)
        if args.is_clip_to_countries:
            arrays = clipcountries(arrays, ref_data=reg_dirs[0])
        gp.rasterize_numpy(arrays, sem_dirs[0], filename=sem_filename)

