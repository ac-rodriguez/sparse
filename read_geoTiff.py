
from __future__ import division
import argparse
import numpy as np
from osgeo import gdal
import os, sys
import fnmatch
import re
import cv2
from collections import defaultdict
from skimage.measure import block_reduce
import glob

import gdal_processing as gp
# from utils import patches
# import patches
run_60 = False

def readHR(args, roi_lon_lat, data_file=None, as_float=True):
    # args = parseargs()
    # roi_lon_lat = args.roi_lon_lat
    if data_file is None:
        data_file = args.HR_file
    print(' [*] Reading HR Data {}'.format(os.path.basename(data_file)))

    dsREF = gdal.Open(data_file)

    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = gp.split_roi_string(roi_lon_lat)
        geo_pts_ref = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]

        if not gp.roi_intersection(dsREF, geo_pts_ref):
            print(" [!] The ROI does not intersect with the data product")
            sys.exit(0)

        xmin, ymin, xmax, ymax = gp.to_xy_box(lims=(roi_lon1, roi_lat1, roi_lon2, roi_lat2), dsREF=dsREF,
                                              enlarge=args.scale)  # we enlarge from 5m to 20m Bands in S2

    else:
        xmin,ymin = 0,0
        xmax,ymax = dsREF.RasterXSize -1, dsREF.RasterYSize-1



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

    for band_id in range(dsREF.RasterCount):
        dsBANDS[band_id] = dsREF.GetRasterBand(band_id+1)

    # x1_0, y1_0 = gp.to_xy(roi_lon1, roi_lat1, dsREF)
    # x2_0, y2_0 = gp.to_xy(roi_lon2, roi_lat2, dsREF)

    # min_coord = np.min([x1_0,y1_0,x2_0,y2_0])
    # xmin_0, xmax_0 = gp.enlarge_pixel(min(x1_0, x2_0), max(x1_0, x2_0), ref = 1)
    # ymin_0, ymax_0 = gp.enlarge_pixel(min(y1_0, y2_0), max(y1_0, y2_0), ref = 1)
    # length_x_0 = xmax_0 - xmin_0 + 1
    # length_y_0 = ymax_0 - ymin_0 + 1

    # complete_image = (length_x_0 == xmax - xmin + 1) & (length_y_0 == ymax - ymin + 1)
    #
    # if min_coord < 0:
    #     view = '-incomplete'
    # elif complete_image:
    #     view = '-complete'
    # else:
    #     view = ''

    # if only_complete_image and (min_coord < 0) and not complete_image:
    #     print(" [!] Roi is not complete")
    #     sys.exit(0)
    if (xmax - xmin + 1 <= 10) or (ymax - ymin + 1 <= 10):
        print(" [!] Roi outside of dataset")
        sys.exit(0)


    data10 = None

    ## Load 10m bands
    # for band_name in range(3):
    for key, val in dsBANDS.iteritems():
        Btemp = val.ReadAsArray(xoff=xmin, yoff=ymin, win_xsize=xmax - xmin + 1, win_ysize=ymax - ymin + 1, buf_xsize=xmax - xmin + 1,
                             buf_ysize=ymax - ymin + 1)
        data10 = np.dstack((data10, Btemp)) if data10 is not None else Btemp


    ## Check if the dataset covers the actual roi with data
    # b3_10_ind = select_bands10.index('B3')
    if dsREF.RasterCount == 3:
        id_ = 2
        chan3 = data10[:, :, id_]
    else:
        id_ = 0
        chan3 = data10
    vis = (chan3 < 1).astype(np.int)
    if np.all(chan3 < 1):
        print(" [!] All data is blank on Band {}".format(id_+1))
        sys.exit(0)
    elif np.sum(vis) > 0:
        print(' [!] The selected image has some blank pixels')
        # sys.exit()
    if as_float:
        return data10.astype(np.float32) / 255.0
    else:
        return  data10
    # patches.save_numpy(data = data10, args=args,folder=str(pixel_size), view=view, filename='data_complete')


    # print(" [*] Success.")


def get_jp2(path,band,res=10):
    if band == 'CLD':
        return glob.glob(path + '/GRANULE/*/QI_DATA/*_CLD_{}m.jp2'.format(res))[0]
    else:
        return glob.glob("{}/GRANULE/*/IMG_DATA/R{}m/*_{}_{}m.jp2".format(path, res,band, res))[0]

def readS2(args, roi_lon_lat):

    data_file = args.LR_file
    # if '_USER_' in data_file:
    #     print("use createPatches_old_format.py to create the patches!")
    #     sys.exit(0)


    # roi_lon_lat = args.roi_lon_lat
    select_bands = args.select_bands


    if data_file.endswith('.xml') or data_file.endswith('.zip'):
        data_file, data_filename = os.path.split(data_file)

    _, folder = os.path.split(data_file)

    dsREFfile = get_jp2(data_file, 'B03', res=10)
    # dsREFfile = os.path.join(data_file, 'geotif', 'Band_B3.tif')
    if not os.path.isfile(dsREFfile):
        print('{} does not exist..'.format(dsREFfile))
        sys.exit(1)

    print(' [*] Reading S2 Data {}'.format(os.path.dirname(os.path.basename(dsREFfile))))

    dsREF = gdal.Open(dsREFfile)

    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = gp.split_roi_string(roi_lon_lat)




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

    xmin, ymin, xmax, ymax = gp.to_xy_box((roi_lon1, roi_lat1, roi_lon2, roi_lat2), dsREF, enlarge=1)

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
    bands10m = {'B02','B03','B04','B08'}
    bands20m = {'B05', 'B06', 'B07', 'B8A','B8B','B11','B12','CLD','SCL'}
    select_bands10 = sorted(list(bands10m & set(select_bands)))
    select_bands20 = sorted(list(bands20m & set(select_bands)))

    dsBANDS = dict()

    # for band_id in select_bands:
    #     dsBANDS[band_id] = gdal.Open(os.path.join(data_file,'geotif','Band_{}.tif'.format(band_id)))
    #     if dsBANDS[band_id] is None:
    #         print(' [!] Band {} is not avaliable'.format(band_id))
    #         sys.exit(1)
    for band_id in select_bands10:
        filename = get_jp2(data_file, band_id, res=10)
        dsBANDS[band_id] = gdal.Open(filename)
        if dsBANDS[band_id] is None:
            raise ValueError(' [!] Band {} is not avaliable'.format(band_id))
    for band_id in select_bands20:
        filename = get_jp2(data_file, band_id, res=20)
        dsBANDS[band_id] = gdal.Open(filename)
        if dsBANDS[band_id] is None:
            raise ValueError(' [!] Band {} is not avaliable'.format(band_id))


    ## Load 10m bands
    for band_name in select_bands10:
        Btemp = dsBANDS[band_name].ReadAsArray(xoff=xmin, yoff=ymin, xsize=xmax - xmin + 1, ysize=ymax - ymin + 1, buf_xsize=xmax - xmin + 1,
                             buf_ysize=ymax - ymin + 1)
        data10 = np.dstack((data10, Btemp)) if data10 is not None else Btemp

    ## Check if the dataset covers the actual roi with data
    b3_10_ind = select_bands10.index('B03')
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



def readS2_old(args, roi_lon_lat):

    # data_type = args.data_type
    data_file = args.LR_file
    # if '_USER_' in data_file:
    #     print("use createPatches_old_format.py to create the patches!")
    #     sys.exit(0)

    # roi_lon_lat = args.roi_lon_lat
    select_bands = args.select_bands

    if data_file.endswith('.xml') or data_file.endswith('.zip'):
        data_file, data_filename = os.path.split(data_file)
    else:
        data_filename = 'MTD_MSIL2A.xml'

    file_date = data_file.split('_')[-1].replace('.SAFE','')
    for file in os.listdir(data_file):
        if fnmatch.fnmatch(file, '*{}.xml'.format(file_date)):
            data_filename = file
            print(file)
    _, folder = os.path.split(data_file)


    raster = gdal.Open(os.path.join(data_file,data_filename))

    if not raster:
        print(" [!] Corrupt 2A XML file...")
        sys.exit(0)
    datasets = raster.GetSubDatasets()

    if not datasets:
        print(" [!] Empty Datasets... Invalid xml format")
        print("     try with createPatches_tif.py...")
        sys.exit(0)
    else:
        tenMsets = []
        twentyMsets = []
        sixtyMsets = []
        unknownMsets = []
        for (dsname, dsdesc) in datasets:
            if '10m resolution' in dsdesc:
                tenMsets += [ (dsname, dsdesc) ]
            elif '20m resolution' in dsdesc:
                twentyMsets += [ (dsname, dsdesc) ]
            elif '60m resolution' in dsdesc:
                sixtyMsets += [ (dsname, dsdesc) ]
            else:
                unknownMsets += [ (dsname, dsdesc) ]

    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [float(x) for x in re.split(',', roi_lon_lat)]
    else:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = -180, -90, 180, 90



    geo_pts_ref = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]
    dsREF = gdal.Open(tenMsets[0][0])
    if not gp.roi_intersection(dsREF, geo_pts_ref):
        print(" [!] The ROI does not intersect with the data product")
        sys.exit(0)

    # case where we have several UTM in the data set
    # => select the one with maximal coverage of the study zone
    utm_idx = 0
    utm = ""
    all_utms =defaultdict(str)
    xmin, ymin, xmax, ymax = 0, 0, 0, 0
    largest_area = -1
    # process even if there is only one 10m set, in order to get roi -> pixels
    for (tmidx, (dsname, dsdesc)) in enumerate(tenMsets + unknownMsets):
        ds = gdal.Open(dsname)

        x1, y1 = gp.to_xy(roi_lon1, roi_lat1, ds)
        x2, y2 = gp.to_xy(roi_lon2, roi_lat2, ds)
        tmxmin = max(min(x1, x2, ds.RasterXSize - 1), 0)
        tmxmax = min(max(x1, x2, 0), ds.RasterXSize - 1)
        tmymin = max(min(y1, y2, ds.RasterYSize - 1), 0)
        tmymax = min(max(y1, y2, 0), ds.RasterYSize - 1)
        # enlarge to the nearest 60 pixel boundary for the super-resolution
        # tmxmin, tmxmax = gp.enlarge_to_60pixel(tmxmin,tmxmax)
        # tmymin, tmymax = enlarge_to_60pixel(tmymin,tmymax)

            # tmxmin = int(tmxmin / 6) * 6
            # tmxmax = int((tmxmax + 1) / 6) * 6 - 1
            # tmymin = int(tmymin / 6) * 6
            # tmymax = int((tmymax + 1) / 6) * 6 - 1

        area = (tmxmax - tmxmin + 1) * (tmymax - tmymin + 1)
        current_utm = dsdesc[dsdesc.find("UTM"):]
        if area > all_utms[current_utm]:
            all_utms[current_utm] = area
        if area > largest_area:
            xmin, ymin, xmax, ymax = tmxmin, tmymin, tmxmax, tmymax
            largest_area = area
            utm_idx = tmidx
            utm = dsdesc[dsdesc.find("UTM"):]

    # convert comma separated band list into a list
    select_bands = set([x for x in re.split(',',select_bands) ])
    select_bands.add('CLD') ## Always get CLD band

    print("Selected UTM Zone: {}".format(utm))
    print("Selected pixel region: xmin=%d, ymin=%d, xmax=%d, ymax=%d:" % (xmin, ymin, xmax, ymax))
    print("Selected pixel region: tmxmin=%d, tmymin=%d, tmxmax=%d, tmymax=%d:" % (tmxmin, tmymin, tmxmax, tmymax))
    print("Image size: width=%d x height=%d" % (xmax - xmin + 1, ymax - ymin + 1))

    if xmax < xmin or ymax < ymin:
        print(" [!] Invalid region of interest / UTM Zone combination")
        sys.exit(0)

    selected_10m_data_set = None
    if not tenMsets:
        selected_10m_data_set = unknownMsets[0]
    else:
        selected_10m_data_set = tenMsets[utm_idx]
    selected_20m_data_set = None
    for (dsname, dsdesc) in enumerate(twentyMsets):
        if utm in dsdesc:
            selected_20m_data_set = (dsname, dsdesc)
    # if not found, assume the listing is in the same order
    # => OK if only one set
    if not selected_20m_data_set: selected_20m_data_set = twentyMsets[utm_idx]
    selected_60m_data_set = None
    for (dsname, dsdesc) in enumerate(sixtyMsets):
        if utm in dsdesc:
            selected_60m_data_set = (dsname, dsdesc)
    if not selected_60m_data_set: selected_60m_data_set = sixtyMsets[utm_idx]

    ds10 = gdal.Open(selected_10m_data_set[0])
    ds20 = gdal.Open(selected_20m_data_set[0])


    if (xmax - xmin + 1 <= 10) or (ymax - ymin + 1 <= 10):
        print(" [!] Roi outside of dataset")
        sys.exit(0)

    def validate_description(description):
        m = re.match("(.*?), central wavelength (\d+) nm", description)
        if m:
            return m.group(1) + " (" + m.group(2) + " nm)"
        # Some HDR restrictions... ENVI band names should not include commas

        pos = description.find(',')
        return description[:pos] + description[(pos + 1):]

    def get_band_short_name(description):
        if ',' in description:
            return description[:description.find(',')]
        if ' ' in description:
            return description[:description.find(' ')]
        return description[:3]

    validated_10m_bands = []
    validated_10m_indices = []
    validated_20m_bands = []
    validated_20m_indices = []
    validated_60m_bands = []
    validated_60m_indices = []
    validated_descriptions = defaultdict(str)
    validated_10m_dict = dict()
    validated_20m_dict = dict()
    sys.stdout.write("Selected 10m bands:")


    for b in range(0, ds10.RasterCount):
        desc = validate_description(ds10.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(desc)
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_10m_bands += [shortname]
            validated_10m_indices += [b]
            validated_descriptions[shortname] = desc
            validated_10m_dict[shortname] = b
    sys.stdout.write("\nSelected 20m bands:")
    for b in range(0, ds20.RasterCount):
        desc = validate_description(ds20.GetRasterBand(b + 1).GetDescription())
        shortname = get_band_short_name(desc)
        if shortname in select_bands:
            sys.stdout.write(" " + shortname)
            select_bands.remove(shortname)
            validated_20m_bands += [shortname]
            validated_20m_indices += [b]
            validated_descriptions[shortname] = desc
            validated_20m_dict[shortname] = b
    sys.stdout.write("\n")

    # add B3 for 20 m resolution

    if validated_10m_indices:
        print("Available {}".format(selected_10m_data_set[1]))
        print("Loading selected 10m Bands: {}".format(validated_10m_bands))
        data10 = np.rollaxis(
            ds10.ReadAsArray(xoff=xmin, yoff=ymin, xsize=xmax - xmin + 1, ysize=ymax - ymin + 1, buf_xsize=xmax - xmin + 1,
                             buf_ysize=ymax - ymin + 1), 0, 3)[:, :, validated_10m_indices]

    b3_10_ind = validated_10m_indices.index(validated_10m_dict['B3'])
    chan3 = data10[:, :, b3_10_ind]
    vis = (chan3 < 1).astype(np.int)
    if np.all(chan3 < 1):
        print(" [!] All data is blank on Band 3")
        sys.exit(0)
    elif np.sum(vis) > 0:
        print(' [!] The selected image has some blank pixels')
        # sys.exit()

    if validated_20m_indices:
        print("Available {}".format(selected_20m_data_set[1]))
        print("Loading selected 20m Bands: {}".format(validated_20m_bands))
        data20 = np.rollaxis(
            ds20.ReadAsArray(xoff=xmin // 2, yoff=ymin // 2, xsize=(xmax - xmin + 1) // 2, ysize=(ymax - ymin + 1) // 2,
                             buf_xsize=(xmax - xmin + 1) // 2, buf_ysize=(ymax - ymin + 1) // 2), 0, 3)[:, :,
                 validated_20m_indices]


    if len(data20.shape) == 2:
        data20 = np.expand_dims(data20, axis = 2)

    return data10.astype(np.float32), data20.astype(np.float32)


def read_labels(args, roi, roi_with_labels, is_HR=False):
    # if args.HR_file is not None:
    ref_scale = 16  # 10m -> 0.625m
    sigma = ref_scale  # /np.pi
    if is_HR:
        ds_file = args.HR_file
        ref_scale = ref_scale // args.scale
        scale_lims = args.scale
    else:
        if 'USER' in args.LR_file:
            ds_file = gp.getrefDataset(args.LR_file, is_use_gtiff=False)
        else:
            ds_file = os.path.join(os.path.dirname(args.LR_file), 'geotif', 'Band_B3.tif')

        scale_lims = 1

    print(' [*] Reading Labels {}'.format(os.path.basename(args.points)))

    ds = gdal.Open(ds_file)
    print(' [*] Reading complete Area')

    lims_H = gp.to_xy_box(roi, ds, enlarge=scale_lims)
    print(' [*] Reading labeled Area')

    lims_with_labels = gp.to_xy_box(roi_with_labels, ds, enlarge=scale_lims)

    labels = gp.rasterize_points_constrained(Input=args.points, refDataset=ds_file, lims=lims_H,
                                             lims_with_labels=lims_with_labels, up_scale=ref_scale,
                                             sigma=sigma, sq_kernel=args.sq_kernel)
    (xmin, ymin, xmax, ymax) = lims_with_labels
    xmin, xmax = xmin - lims_H[0], xmax - lims_H[0]
    ymin, ymax = ymin - lims_H[1], ymax - lims_H[1]
    return np.expand_dims(labels, axis=2), (xmin, ymin, xmax, ymax)


def read_labels_semseg(args, sem_file,dsm_file, is_HR, ref_scale=16):
    # if args.HR_file is not None:
    # ref_scale = 16  # 10m -> 0.625m
    # ref_scale = ref_scale // args.scale

    print(' [*] Reading Labels {}'.format(os.path.basename(sem_file)))

    # ds = gdal.Open(ds_file)
    print(' [*] Reading complete Area')

    # lims_H = gp.to_xy_box(roi, ds, enlarge=scale_lims)
    print(' [*] Reading labeled Area')

    # lims_with_labels = gp.to_xy_box(roi_with_labels, ds, enlarge=scale_lims)

    labels = readHR(args, roi_lon_lat=None, data_file=sem_file, as_float=False)
    lut = np.ones(256, dtype=np.uint8) * 255
    lut[[255, 29, 179, 150, 226, 76]] = np.arange(6, dtype=np.uint8)
    labels = cv2.LUT(cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY), lut)
    # 3 vegetation
    # 5 buildings
    # 255 with or without gt

    if is_HR: ref_scale = ref_scale //args.scale

    labels = block_reduce(labels,(ref_scale,ref_scale),np.median)

    labels = labels.astype(np.float32)
    mask_out = labels == 255.0
    labels[mask_out] = -1.

    labels_reg = readHR(args, roi_lon_lat=None, data_file=dsm_file, as_float=False)
    labels_reg = labels_reg.astype(np.float32)

    labels_reg = block_reduce(labels_reg,(ref_scale,ref_scale),np.mean)

    labels_reg[mask_out] = -1.

    im_out = np.stack((labels,labels_reg), axis=-1)

    xmin, xmax = 0, im_out.shape[1]
    ymin, ymax = 0, im_out.shape[1]
    return im_out,(xmin, ymin, xmax, ymax)
