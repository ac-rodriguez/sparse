import os
import sys
import glob
import numpy as np

from gdal_merge import main as untile
from osgeo import gdal, gdalconst
from gdal_calc import main as gdal_clip

ref_band = 1
#
# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmriau_simpleA9all'
# ref_proj = 'EPSG:32647' # UTM 47N Riau


# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmpeninsulanew_simpleA9all'
# ref_proj = 'EPSG:32647' # UTM 47N Peninsula

# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmsabah_simpleA9all'
# ref_proj = 'EPSG:32650' # UTM 50N SABAH


# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmsarawak3_simpleA20clean_all'
# ref_proj = 'EPSG:32649' # UTM 49N SARAWAK


# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmsarawak3_simpleA20westkalim'
data_dir = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmcocotiles2_simpleA'
ref_proj = 'EPSG:32749' # UTM 49S SARAWAK
ref_band = 2


is_overwrite = False

filepaths = glob.glob(data_dir+'/*_preds_reg.tif')
print('number of files: {}'.format(len(filepaths)))

ref_proj_name = ref_proj.replace(':','_')
clip_val = 0.02

for file in filepaths:
    print(file)

    # clip
    f_clip = file.replace('.tif', f'_clip{clip_val}.tif')
    if not os.path.isfile(f_clip) or is_overwrite:
        # gdal_clip(["-A", file, f"--outfile={f_clip}", "--calc=\"A*(A>0.01)\"", "--NoDataValue=0"])
        os.system(f'gdal_calc.py -A {file} --outfile={f_clip} --calc=\"A*(A>{clip_val})\" --NoDataValue=0 --A_band={ref_band}')

for down_scale in [10,20,400]: # 10,20,

    filepaths_wgs84 = []
    for file in filepaths:
        print(file)

        # clip
        f_clip = file.replace('.tif', f'_clip{clip_val}.tif')
        # project to WGS 84
        f_down = f_clip.replace('.tif', f'_down{down_scale}.vrt')
        ds = gdal.Open(f_clip, gdalconst.GA_ReadOnly)

        geot = ds.GetGeoTransform()

        # filepaths_wgs84.append(f_down)

        if not os.path.isfile(f_down) or is_overwrite:
            src_ds = gdal.Open(f_clip, gdalconst.GA_ReadOnly)
            #
            warp_opts = gdal.WarpOptions(
                format="VRT",  # format='GTiff',
                resampleAlg=gdalconst.GRA_Average, xRes=geot[1]*down_scale, yRes=geot[5]*down_scale,
                srcNodata=99,
                dstNodata='99')
            gdal.Warp(f_down, src_ds, options=warp_opts)
            print('{} saved!'.format(f_down))


        # project to WGS 84
        f_projected = f_down.replace('.vrt', f'_{ref_proj_name}.vrt')

        filepaths_wgs84.append(f_projected)

        if not os.path.isfile(f_projected) or is_overwrite:
            # src_ds = gdal.Open(file, gdalconst.GA_ReadOnly)

            warp_opts = gdal.WarpOptions(
                format="VRT",  # format='GTiff',
                dstSRS=ref_proj,
                resampleAlg=gdalconst.GRA_Bilinear,
                # srcNodata=99,
                dstNodata='nan')
            gdal.Warp(f_projected, f_down, options=warp_opts)
            print('{} saved!'.format(f_projected))



    # merge all tiles in WGS 84
    print('untiling...')

    untile(names=filepaths_wgs84, out_file=f'{data_dir}/0_{ref_proj_name}_down{down_scale}.tif')

    print('DONE!')
