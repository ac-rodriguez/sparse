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
from utils.data_reader import read_and_upsample_sen2

parser = argparse.ArgumentParser(description="median per tile calculation",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("data_dir", type=str, default="",
                    help="Path to the directory containing the input SAFE files")
parser.add_argument("tile", type=str, default="",
                    help="Tile name")

parser.add_argument("--save-dir", default=None, help='Default will save in parent directory of data_dir')
parser.add_argument("--is-overwrite", default=False, action="store_true",
                    help="overwrite predictions already in folder")
parser.add_argument("--compression", default='12', help='compression algorithm to save geotifs')
parser.add_argument("--function", default="mean",choices=["mean","median"], help='compression algorithm to save geotifs')

args = parser.parse_args()


if 'pf-pc' in socket.gethostname():
    PATH = '/scratch/andresro/leon_work'
else:
    PATH = '/cluster/work/igp_psr/andresro'


if args.save_dir is None:
    args.save_dir = os.path.dirname(args.data_dir)
if not os.path.exists(args.save_dir):    os.makedirs(args.save_dir)

# tilename = os.path.basename(args.data_dir)
ref_tile = args.tile


fileformat = '.zip'
suffix = args.function


mask_dict = {'CLD':10, # if Cloud prob is higher than 10%
            'SCL':[3,11], # if SCL is equal to cloud, cloud shadow
            '20m':0} # if all 20m bands are 0


list_dirs = glob.glob(f'{args.data_dir}/*{ref_tile}*{fileformat}')


save_filename = f'{args.save_dir}/{ref_tile}_{suffix}.tif'


if not os.path.isfile(save_filename) or args.is_overwrite:
    # is_low_mem = True

    if args.function == "median":
        arrays = []
        for file in list_dirs:
            print('reading',list_dirs.index(file), os.path.basename(file))
            data_  = read_and_upsample_sen2(file,args=None,roi_lon_lat=None,mask_out_dict=mask_dict,is_skip_if_masked=False)
            arrays.append(data_)
            
        arrays = np.stack(arrays,axis=0)
        arrays = np.nanmedian(arrays,axis=0)

        gp.rasterize_numpy(arrays,refDataset=list_dirs[0],filename=save_filename,type='float32', compression=args.compression)

    else:
        arrays = None
        n = 0
        for file in list_dirs:
            print('reading',list_dirs.index(file), os.path.basename(file))
            data_  = read_and_upsample_sen2(file,args=None,roi_lon_lat=None,mask_out_dict=mask_dict,is_skip_if_masked=False, verbose=False)

            valid_pixels = 1-np.isnan(data_[...,0])
            n = n + (valid_pixels)
            arrays = np.nansum(np.stack((arrays,data_),axis=0),axis=0) if arrays is not None else data_

        print('shapes: arrays, n')
        print(arrays.shape,n.shape)
        means = arrays/n 

        gp.rasterize_numpy(means,refDataset=list_dirs[0],filename=save_filename,type='float32', compression=args.compression)

