
from __future__ import division
import argparse
import numpy as np
from osgeo import gdal
import os, sys

import re

import gdal_processing as gp
# from utils import patches
# import patches
run_60 = False

def readHR(args, roi_lon_lat):
    # args = parseargs()
    # roi_lon_lat = args.roi_lon_lat

    data_file = args.HR_file
    dsREF = gdal.Open(data_file)


    # pixel_size = dsREF.GetGeoTransform()[1]


    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [float(x) for x in re.split(',', roi_lon_lat)]
    else:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = -180, -90, 180, 90


    geo_pts_ref = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]


    if not gp.roi_intersection(dsREF, geo_pts_ref):
        print(" [!] The ROI does not intersect with the data product")
        sys.exit(0)


    xmin, ymin, xmax, ymax = gp.to_xy_box(lims=(roi_lon1, roi_lat1, roi_lon2, roi_lat2),dsREF= dsREF, enlarge=4) # we enlarge from 5m to 20m Bands in S2

    # elif not roi_lon_lat:
    #     tmxmin = 0
    #     tmxmax = dsREF.RasterXSize - 1
    #     tmymin = 0
    #     tmymax = dsREF.RasterYSize - 1
    # else:
    #
    #     x1, y1 = gp.to_xy(roi_lon1, roi_lat1, dsREF)
    #     x2, y2 = gp.to_xy(roi_lon2, roi_lat2, dsREF)
    #     tmxmin = max(min(x1, x2, dsREF.RasterXSize - 1), 0)
    #     tmxmax = min(max(x1, x2, 0), dsREF.RasterXSize - 1)
    #     tmymin = max(min(y1, y2, dsREF.RasterYSize - 1), 0)
    #     tmymax = min(max(y1, y2, 0), dsREF.RasterYSize - 1)
    #     # enlarge to the nearest 60 pixel boundary for the super-resolution
    #     if False:
    #         tmxmin, tmxmax = gp.enlarge_pixel(tmxmin, tmxmax, ref = 1)
    #         tmymin, tmymax = gp.enlarge_pixel(tmymin, tmymax, ref = 1)
    #
    # xmin, ymin, xmax, ymax = tmxmin, tmymin, tmxmax, tmymax
    utm = 'NaN'


    print("Selected UTM Zone: {}".format(utm))
    print("Selected pixel region: xmin=%d, ymin=%d, xmax=%d, ymax=%d:" % (xmin, ymin, xmax, ymax))
    # print("Selected pixel region: tmxmin=%d, tmymin=%d, tmxmax=%d, tmymax=%d:" % (tmxmin, tmymin, tmxmax, tmymax))
    print("Image size: width=%d x height=%d" % (xmax - xmin + 1, ymax - ymin + 1))

    if xmax < xmin or ymax < ymin:
        print(" [!] Invalid region of interest / UTM Zone combination")
        sys.exit(0)

    dsBANDS = dict()

    for band_id in range(3):
        dsBANDS[band_id] = dsREF.GetRasterBand(band_id+1)

    x1_0, y1_0 = gp.to_xy(roi_lon1, roi_lat1, dsREF)
    x2_0, y2_0 = gp.to_xy(roi_lon2, roi_lat2, dsREF)

    min_coord = np.min([x1_0,y1_0,x2_0,y2_0])
    xmin_0, xmax_0 = gp.enlarge_pixel(min(x1_0, x2_0), max(x1_0, x2_0), ref = 1)
    ymin_0, ymax_0 = gp.enlarge_pixel(min(y1_0, y2_0), max(y1_0, y2_0), ref = 1)
    length_x_0 = xmax_0 - xmin_0 + 1
    length_y_0 = ymax_0 - ymin_0 + 1

    complete_image = (length_x_0 == xmax - xmin + 1) & (length_y_0 == ymax - ymin + 1)

    if min_coord < 0:
        view = '-incomplete'
    elif complete_image:
        view = '-complete'
    else:
        view = ''

    # if only_complete_image and (min_coord < 0) and not complete_image:
    #     print(" [!] Roi is not complete")
    #     sys.exit(0)
    if (xmax - xmin + 1 <= 10) or (ymax - ymin + 1 <= 10):
        print(" [!] Roi outside of dataset")
        sys.exit(0)


    data10 = None

    ## Load 10m bands
    for band_name in range(3):
        Btemp = dsBANDS[band_name].ReadAsArray(xoff=xmin, yoff=ymin, win_xsize=xmax - xmin + 1, win_ysize=ymax - ymin + 1, buf_xsize=xmax - xmin + 1,
                             buf_ysize=ymax - ymin + 1)
        data10 = np.dstack((data10, Btemp)) if data10 is not None else Btemp


    ## Check if the dataset covers the actual roi with data
    # b3_10_ind = select_bands10.index('B3')
    chan3 = data10[:, :, 2]
    vis = (chan3 < 1).astype(np.int)
    if np.all(chan3 < 1):
        print(" [!] All data is blank on Band 3")
        sys.exit(0)
    elif np.sum(vis) > 0:
        print(' [!] The selected image has some blank pixels')
        # sys.exit()

    return data10.astype(np.float32) / 255.0
    # patches.save_numpy(data = data10, args=args,folder=str(pixel_size), view=view, filename='data_complete')


    # print(" [*] Success.")


def readS2(args, roi_lon_lat):

    data_file = args.LR_file
    if '_USER_' in data_file:
        print("use createPatches_old_format.py to create the patches!")
        sys.exit(0)


    # roi_lon_lat = args.roi_lon_lat
    select_bands = args.select_bands


    if data_file.endswith('.xml') or data_file.endswith('.zip'):
        data_file, data_filename = os.path.split(data_file)

    _, folder = os.path.split(data_file)


    dsREFfile = os.path.join(data_file, 'geotif', 'Band_B3.tif')
    if not os.path.isfile(dsREFfile):
        print('{} does not exist..'.format(dsREFfile))
        sys.exit(1)
    dsREF = gdal.Open(dsREFfile)

    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [float(x) for x in re.split(',', roi_lon_lat)]




        # x1, y1 = gp.to_xy(roi_lon1, roi_lat1, dsREF)
        # x2, y2 = gp.to_xy(roi_lon2, roi_lat2, dsREF)
        # tmxmin = max(min(x1, x2, dsREF.RasterXSize - 1), 0)
        # tmxmax = min(max(x1, x2, 0), dsREF.RasterXSize - 1)
        # tmymin = max(min(y1, y2, dsREF.RasterYSize - 1), 0)
        # tmymax = min(max(y1, y2, 0), dsREF.RasterYSize - 1)

    else:

        # roi_x1, roi_y1, roi_x2, roi_y2 =0,dsREF.RasterXSize+1,0,dsREF.RasterYSize +1
        #
        # tmxmin = max(min(roi_x1, roi_x2, dsREF.RasterXSize - 1), 0)
        # tmxmax = min(max(roi_x1, roi_x2, 0), dsREF.RasterXSize - 1)
        # tmymin = max(min(roi_y1, roi_y2, dsREF.RasterYSize - 1), 0)
        # tmymax = min(max(roi_y1, roi_y2, 0), dsREF.RasterYSize - 1)

        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = -180, -90, 180, 90

    xmin, ymin, xmax, ymax = gp.to_xy_box((roi_lon1, roi_lat1, roi_lon2, roi_lat2), dsREF)

    # xmin, ymin, xmax, ymax = tmxmin, tmymin, tmxmax, tmymax
    utm = 'NaN'

    geo_pts_ref = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]


    # dsREF = gdal.Open(tenMsets[0][0])
    if not gp.roi_intersection(dsREF, geo_pts_ref):
        print(" [!] The ROI does not intersect with the data product")
        sys.exit(0)

    # convert comma separated band list into a list
    select_bands = set([x for x in re.split(',',select_bands)])
    select_bands.add('CLD') ## Always get CLD band

    print("Image size: width={} x height={}".format(xmax - xmin + 1, ymax - ymin + 1))

    if xmax < xmin or ymax < ymin:
        print(" [!] Invalid region of interest / UTM Zone combination")
        sys.exit(0)

    if (xmax - xmin + 1 <= 10) or (ymax - ymin + 1 <= 10):
        print(" [!] Roi outside of dataset")
        sys.exit(0)


    data10 = data20 = None
    bands10m = {'B2','B3','B4','B8'}
    bands20m = {'B5', 'B6', 'B7', 'B8A','B8B','B11','B12','CLD','SCL'}
    select_bands10 = sorted(list(bands10m & set(select_bands)))
    select_bands20 = sorted(list(bands20m & set(select_bands)))

    dsBANDS = dict()

    for band_id in select_bands:
        dsBANDS[band_id] = gdal.Open(os.path.join(data_file,'geotif','Band_{}.tif'.format(band_id)))
        if dsBANDS[band_id] is None:
            print(' [!] Band {} is not avaliable'.format(band_id))
            sys.exit(1)

    ## Load 10m bands
    for band_name in select_bands10:
        Btemp = dsBANDS[band_name].ReadAsArray(xoff=xmin, yoff=ymin, xsize=xmax - xmin + 1, ysize=ymax - ymin + 1, buf_xsize=xmax - xmin + 1,
                             buf_ysize=ymax - ymin + 1)
        data10 = np.dstack((data10, Btemp)) if data10 is not None else Btemp

    ## Check if the dataset covers the actual roi with data
    b3_10_ind = select_bands10.index('B3')
    chan3 = data10[:, :, b3_10_ind]
    vis = (chan3 < 1).astype(np.int)
    if np.all(chan3 < 1):
        print(" [!] All data is blank on Band 3")
        sys.exit(0)
    elif np.sum(vis) > 0:
        print(' [!] The selected image has some blank pixels')
        # sys.exit()

    ## Load 20m bands
    for band_name in select_bands20:
        Btemp = dsBANDS[band_name].ReadAsArray(xoff=xmin // 2, yoff=ymin // 2, xsize=(xmax - xmin + 1) // 2, ysize=(ymax - ymin + 1) // 2,
                             buf_xsize=(xmax - xmin + 1) // 2, buf_ysize=(ymax - ymin + 1) // 2)
        data20 = np.dstack((data20, Btemp)) if data20 is not None else Btemp

    print("Selected 10m bands: {}".format(select_bands10))
    print("Selected 20m bands: {}".format(select_bands20))


    if len(data20.shape) == 2:
        data20 = np.expand_dims(data20, axis = 2)

    return data10.astype(np.float32), data20.astype(np.float32)
    # patches.save_numpy(data10, data20, labels, select_bands10, select_bands20, args, folder, view, filename='data', points = points)
    # print(" [*] Success.")


