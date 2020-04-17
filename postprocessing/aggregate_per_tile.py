import os, sys
import gdal
import argparse
import socket
import numpy as np
import glob
from scipy import stats
from skimage.transform import resize

sys.path.insert(0, os.path.dirname(os.getcwd()))
sys.path.insert(0, os.getcwd())
import utils.gdal_processing as gp


parser = argparse.ArgumentParser(description="median per tile calculation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir", type=str, default="",
                    help="Path to the directory containing the predicted SAFE files")
parser.add_argument("--save-dir", default=None, help='Default will save in parent directory of data_dir')
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="overwrite predictions already in folder")
parser.add_argument("--is-clip-to-countries", default=False, action="store_true",
                    help="clip values outside landdefined areas")
parser.add_argument("--is-reg-only", default=False, action="store_true",
                    help=" compute regression output only (omit semantic mask)")
parser.add_argument("--is-remove-water", default=False, action="store_true",
                    help=" if any SCL in the tile is water set the prediction to 0")
parser.add_argument("--is-avgprobs", default=False, action="store_true",
                    help="if true: average probabilities and then compute argmax, else mayority voting of classes")
parser.add_argument("--nan-class", default=99, type=int,
                    help="Value of output pixels with nan values in all the input datasets")
parser.add_argument("--mc-repetitions", default=1, type=int,
                    help="Number of repetitions the arrays have predictions from")
# parser.add_argument("--n-class", default=1, type=int,
#                     help="Number of predicted classes")
args = parser.parse_args()


def nanargmax(data, nanclass=args.nan_class):
    # nanargmax does not handle properly pixels with all Nan values
    mask = np.isnan(data)
    mask = np.all(mask, axis=-1)

    data[mask] = -np.inf

    classes = np.nanargmax(data, axis=-1)
    classes[mask] = nanclass
    return classes

def clean_array(x, is_water, ref_data):
    x[np.isnan(x)] = args.nan_class
    if args.is_remove_water:
        x[is_water] = 0
    if args.is_clip_to_countries:
        x = clipcountries(x,ref_data=ref_data)
    return x

if 'pf-pc' in socket.gethostname():
    PATH = '/scratch/andresro/leon_work'
else:
    PATH = '/cluster/work/igp_psr/andresro'

def clipcountries(arrays, ref_data):
    shpfile = PATH + '/barry_palm/data/labels/countries/3countries.shp'
    shp_raster = gp.rasterize_polygons(shpfile, ref_data, as_bool=True)
    arrays[~shp_raster] = args.nan_class

    return arrays


if args.save_dir is None:
    args.save_dir = os.path.dirname(args.data_dir)
if not os.path.exists(args.save_dir):    os.makedirs(args.save_dir)

tilename = os.path.basename(args.data_dir)

if args.is_remove_water:
    path1 = PATH+'/barry_palm/data/2A/palmcountries_2017/'
    list1 = glob.glob(f'{path1}/*{tilename}*2017*.SAFE')

    is_water = []
    # valid_pixels = []
    for count, file in enumerate(list1):
        #     if len(id_) > 0:
        scl_name = glob.glob(file + "/GRANULE/*/IMG_DATA/R20m/*_SCL_20m.jp2")
        if len(scl_name) > 0:
            ds = gdal.Open(scl_name[0])
            array_cld = ds.ReadAsArray()
            array_cld = array_cld.transpose().swapaxes(0, 1)
            array_cld = array_cld == 6

            is_water.append(array_cld)
    is_water = np.stack(is_water, axis=-1)

    is_water = np.nansum(is_water, axis=-1) > 0
    is_water = resize(is_water, output_shape=[x*2 for x in is_water.shape[0:2]], mode='edge') > 0.5
else:
    is_water = None

suffix_reg = f'{args.mc_repetitions}_preds_reg_0'
if tilename.startswith('T'):
    # Add all the orbits in the tile
    path = os.path.dirname(args.data_dir)
    reg_dirs = glob.glob(f'{path}/*{tilename}/*20170422T024551*{suffix_reg}.tif')
    print(f'reading {len(reg_dirs)} dirs from {path}/*{tilename}')
else:
    reg_dirs = glob.glob(args.data_dir+'/*{suffix_reg}.tif')
    print(f'reading {len(reg_dirs)} dirs from {args.data_dir}')


reg_filename = f'{args.save_dir}/{tilename}_{suffix_reg}.tif'
if not os.path.isfile(reg_filename) or args.is_overwrite:
    is_low_mem = True

    if args.mc_repetitions == 1:
        arrays = []
        for file in reg_dirs:
            print('reading', os.path.basename(file))
            ds = gdal.Open(file)
            nbands = ds.RasterCount
            data_ = [ds.GetRasterBand(x+1).ReadAsArray() for x in range(nbands)]
            data_ = np.stack(data_, axis=-1)
            arrays.append(data_)

        arrays = np.stack(arrays, axis=-1)
        print('computing median value')
        medians = np.nanmedian(arrays, axis=-1)
        medians = clean_array(medians,is_water,reg_dirs[0])
        gp.rasterize_numpy(medians,reg_dirs[0],filename=reg_filename,type='float32')

    if args.mc_repetitions > 1:
        if not is_low_mem:
            arrays = []
            for file in reg_dirs:
                print('reading',reg_dirs.index(file), os.path.basename(file))
                ds = gdal.Open(file)
                nbands = ds.RasterCount
                data_ = [ds.GetRasterBand(x+1).ReadAsArray() for x in range(nbands)]
                data_ = np.stack(data_, axis=-1)
                arrays.append(data_)

            arrays = np.stack(arrays, axis=-2)
            n = np.count_nonzero(~np.isnan(arrays[...,0]), axis=-1) * args.mc_repetitions          
            arrays = np.nansum(arrays,axis=-2)
        else:
            arrays = None
            n = 0
            for file in reg_dirs:
                print('reading',reg_dirs.index(file), os.path.basename(file))
                ds = gdal.Open(file)
                nbands = ds.RasterCount
                data_ = [ds.GetRasterBand(x+1).ReadAsArray() for x in range(nbands)]
                data_ = np.stack(data_, axis=-1)

                n = n + (1-np.isnan(data_[...,0])) * args.mc_repetitions
                arrays = np.nansum(np.stack((arrays,data_),axis=0),axis=0) if arrays is not None else data_

        gp.rasterize_numpy(n,reg_dirs[0],filename=reg_filename.replace('.tif','_n.tif'),type='int32', options=1.2)
            
        x_sum = arrays[...,0]
        x2_sum = arrays[...,1]

        means = x_sum/n 
        means = clean_array(means,is_water,reg_dirs[0])
        gp.rasterize_numpy(means,reg_dirs[0],filename=reg_filename,type='float32', options=1.2)

        std_dev = (x2_sum / n - (x_sum/n)**2)
        std_dev = clean_array(std_dev,is_water,reg_dirs[0])
        gp.rasterize_numpy(std_dev,reg_dirs[0],filename=reg_filename.replace('.tif','_var.tif'),type='float32', options=1.2)
        
        std_dev = (x2_sum / n - (x_sum/n)**2)**0.5
        std_dev = clean_array(std_dev,is_water,reg_dirs[0])
        gp.rasterize_numpy(std_dev,reg_dirs[0],filename=reg_filename.replace('.tif','_std.tif'),type='float32', options=1.2)



classprob_filename = f'{args.save_dir}/{tilename}_preds_classprob.tif'
is_compute_class = not os.path.isfile(classprob_filename) or args.is_overwrite

sem_filename = f'{args.save_dir}/{tilename}_preds_sem.tif'

if args.is_avgprobs:
    sem_filename = sem_filename.replace('sem.tif','semA.tif')
    is_compute_class = False

is_compute_sem = (not os.path.isfile(sem_filename) or args.is_overwrite)

if (is_compute_class or is_compute_sem) and not args.is_reg_only:
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
            if args.is_remove_water:
                arrays[is_water] = 0
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
        if args.is_remove_water:
            arrays[is_water] = 0 # TODO check if 0 is really background class
        if args.is_clip_to_countries:
            arrays = clipcountries(arrays, ref_data=reg_dirs[0])
        gp.rasterize_numpy(arrays, sem_dirs[0], filename=sem_filename)

