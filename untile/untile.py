import os
import sys
import glob
import numpy as np

from gdal_merge import main as untile
from osgeo import gdal, gdalconst
from gdal_calc import main as gdal_clip





# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmpeninsulanew_simpleA9all'

# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmsabah_simpleA9all'
data_dir = '/scratch/andresro/leon_work/sparse/inference/palmsarawak_simpleA20_allsarawak'
ref_proj = 'EPSG:32649' # UTM 49N


# data_dir = '/scratch/andresro/leon_igp/sparse/inference/palmsarawak3_simpleA20clean_all'


filepaths = glob.glob(data_dir+'/*_preds_reg.tif')
print('number of files: {}'.format(len(filepaths)))

for down_scale in [20]: # 10,20,

    clip_val = 0.02
    filepaths_wgs84 = []
    for file in filepaths:
        print(file)

        # clip
        f_clip = file.replace('.tif', f'_clip{clip_val}.tif')
        if not os.path.isfile(f_clip):

            # gdal_clip(["-A", file, f"--outfile={f_clip}", "--calc=\"A*(A>0.01)\"", "--NoDataValue=0"])
            os.system(f'gdal_calc.py -A {file} --outfile={f_clip} --calc=\"A*(A>{clip_val})\" --NoDataValue=0')

        # project to WGS 84
        f_down = f_clip.replace('.tif', f'_down{down_scale}.vrt')
        ds = gdal.Open(f_clip, gdalconst.GA_ReadOnly)

        geot = ds.GetGeoTransform()

        # filepaths_wgs84.append(f_down)

        if not os.path.isfile(f_down):
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
        f_wgs84 = f_down.replace('.vrt', f'_EPSG_4326.vrt')

        filepaths_wgs84.append(f_wgs84)

        if not os.path.isfile(f_wgs84):
            # src_ds = gdal.Open(file, gdalconst.GA_ReadOnly)

            warp_opts = gdal.WarpOptions(
                format="VRT",  # format='GTiff',
                dstSRS='EPSG:4326',
                resampleAlg=gdalconst.GRA_Bilinear,
                # srcNodata=99,
                dstNodata='nan')
            gdal.Warp(f_wgs84, f_down, options=warp_opts)
            print('{} saved!'.format(f_wgs84))



    # merge all tiles in WGS 84
    print('untiling...')

    untile(names=filepaths_wgs84, out_file=f'{data_dir}/0_untiled_down{down_scale}.tif')

    print('DONE!')
