
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

run_60 = False

def readHR(roi_lon_lat, data_file, scale, as_float=True):
    if data_file is None:
        return None
    print(' [*] Reading HR Data {}'.format(os.path.basename(data_file)))

    dsREF = gdal.Open(data_file)

    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = gp.split_roi_string(roi_lon_lat)
        geo_pts_ref = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]

        if not gp.roi_intersection(dsREF, geo_pts_ref):
            print(" [!] The ROI does not intersect with the data product")
            sys.exit(0)

        xmin, ymin, xmax, ymax = gp.to_xy_box(lims=(roi_lon1, roi_lat1, roi_lon2, roi_lat2), dsREF=dsREF,
                                              enlarge=scale)  # we enlarge from 5m to 20m Bands in S2

        print("Selected pixel region: xmin=%d, ymin=%d, xmax=%d, ymax=%d:" % (xmin, ymin, xmax, ymax))
        print("Image size: width=%d x height=%d" % (xmax - xmin + 1, ymax - ymin + 1))

    else:
        xmin,ymin = 0,0
        xmax,ymax = dsREF.RasterXSize -1, dsREF.RasterYSize-1


    if xmax < xmin or ymax < ymin:
        print(" [!] Invalid region of interest / UTM Zone combination")
        sys.exit(0)

    dsBANDS = dict()

    for band_id in range(dsREF.RasterCount):
        dsBANDS[band_id] = dsREF.GetRasterBand(band_id+1)

    if (xmax - xmin + 1 <= 10) or (ymax - ymin + 1 <= 10):
        raise Exception(" [!] Roi outside of dataset")


    data10 = None

    ## Load 10m bands
    # for band_name in range(3):
    for key, val in dsBANDS.items():
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


def readS2(args, roi_lon_lat, data_file=None, is_get_SCL=False):
    if data_file is None: data_file = args.LR_file
    # data_file = args.LR_file
    # if '_USER_' in data_file:
    #     print("use createPatches_old_format.py to create the patches!")
    #     sys.exit(0)


    # roi_lon_lat = args.roi_lon_lat
    select_bands = args.select_bands


    if data_file.endswith('.xml') or data_file.endswith('.zip'):
        data_file, data_filename = os.path.split(data_file)

    _, folder = os.path.split(data_file)

    dsREFfile = gp.get_jp2(data_file, 'B03', res=10)
    # dsREFfile = os.path.join(data_file, 'geotif', 'Band_B3.tif')
    if not os.path.isfile(dsREFfile):
        raise ValueError('{} does not exist..'.format(dsREFfile))

    print(' [*] Reading S2 Data {}'.format(folder))

    dsREF = gdal.Open(dsREFfile)

    if roi_lon_lat:

        if not gp.roi_intersection(dsREF, roi_lon_lat):
            print(" [!] The ROI does not intersect with the data product, skipping it")
            return None, None
        xmin, ymin, xmax, ymax = gp.to_xy_box(roi_lon_lat, dsREF, enlarge=1)

    else:
        xmin, ymin = 0, 0
        xmax, ymax = dsREF.RasterXSize - 1, dsREF.RasterYSize - 1

    # convert comma separated band list into a list
    select_bands = set([x for x in re.split(',',select_bands)])
    select_bands.add('CLD') ## Always get CLD band
    if is_get_SCL:
        select_bands.add('SCL') ## Always get CLD band

    print("Image size: width={} x height={}".format(xmax - xmin + 1, ymax - ymin + 1))

    if xmax < xmin or ymax < ymin:
        print(" [!] Invalid region of interest / UTM Zone combination")
        sys.exit(0)

    if (xmax - xmin + 1 <= 10) or (ymax - ymin + 1 <= 10):
        raise ValueError(" [!] Roi outside of dataset")


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
        filename = gp.get_jp2(data_file, band_id, res=10)
        dsBANDS[band_id] = gdal.Open(filename)
        if dsBANDS[band_id] is None:
            raise ValueError(' [!] Band {} is not avaliable'.format(band_id))
    for band_id in select_bands20:
        filename = gp.get_jp2(data_file, band_id, res=20)
        dsBANDS[band_id] = gdal.Open(filename)
        if dsBANDS[band_id] is None:
            raise ValueError(' [!] Band {} is not avaliable'.format(band_id))

    ## Load 20m bands
    for band_name in select_bands20:
        Btemp = dsBANDS[band_name].ReadAsArray(xoff=xmin // 2, yoff=ymin // 2, xsize=(xmax - xmin + 1) // 2,
                                               ysize=(ymax - ymin + 1) // 2,
                                               buf_xsize=(xmax - xmin + 1) // 2, buf_ysize=(ymax - ymin + 1) // 2)
        if band_name == 'CLD':
            cloudy = np.mean(Btemp > 50)
            if cloudy > 0.5:
                print(f' [!] Dataset with P_cloud > 0.5 in {cloudy*100:.2f}% of the pixels, skipping it...')
                return None, None


        data20 = np.dstack((data20, Btemp)) if data20 is not None else Btemp

    print("Selected 10m bands: {}".format(select_bands10))
    print("Selected 20m bands: {}".format(select_bands20))


    if len(data20.shape) == 2:
        data20 = np.expand_dims(data20, axis=2)

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
        print(" [!] All data is blank on Band 3, returning None, None")
        return None, None
        # sys.exit(0)
    elif np.sum(vis) > 0:
        print(' [!] The selected image has some blank pixels')
        # sys.exit()


    return data10.astype(np.float32), data20.astype(np.float32)
    # patches.save_numpy(data10, data20, labels, select_bands10, select_bands20, args, folder, view, filename='data', points = points)
    # print(" [*] Success.")



def readS2_old(args, roi_lon_lat, data_file =None):
    if data_file is None: data_file = args.LR_file

    # data_type = args.data_type
    # data_file = args.LR_file
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


def read_labels(args,shp_file, roi, roi_with_labels, ref_hr=None, ref_lr=None, is_HR=False,):
    if shp_file is None:
        return None,None
    ref_scale = 16  # 10m -> 0.625m
    sigma = ref_scale  # /np.pi
    if is_HR:
        ds_file = ref_hr
        ref_scale = ref_scale // args.scale
        scale_lims = args.scale
    else:
        lr_file = ref_lr
        if 'USER' in lr_file:
            ds_file = gp.getrefDataset(lr_file, is_use_gtiff=False)
        else:
            if lr_file.endswith('.xml'):
                lr_file = os.path.dirname(lr_file)
            ds_file = gp.get_jp2(lr_file, 'B03', res=10)

        scale_lims = 1

    print(' [*] Reading Labels {}'.format(os.path.basename(shp_file)))

    ds = gdal.Open(ds_file)
    print(' [*] Reading complete Area')

    lims_H = gp.to_xy_box(roi, ds, enlarge=scale_lims)
    print(' [*] Reading labeled Area')

    lims_with_labels = gp.to_xy_box(roi_with_labels, ds, enlarge=scale_lims)
    if shp_file.endswith('.shp'):
        labels = gp.rasterize_polygons(InputVector=shp_file,refDataset=ds_file,lims=lims_with_labels,attribute=args.attr)
    elif shp_file.endswith('.tif'):
        labels = readHR(data_file=shp_file, roi_lon_lat=roi_with_labels, scale=args.scale)
    else:
        labels = gp.rasterize_points_constrained(Input=shp_file, refDataset=ds_file, lims=lims_H,
                                                 lims_with_labels=lims_with_labels, up_scale=ref_scale,
                                                 sigma=sigma, sq_kernel=args.sq_kernel)
    (xmin, ymin, xmax, ymax) = lims_with_labels
    xmin, xmax = xmin - lims_H[0], xmax - lims_H[0]
    ymin, ymax = ymin - lims_H[1], ymax - lims_H[1]
    return np.expand_dims(labels, axis=2), (xmin, ymin, xmax, ymax)


def read_labels_semseg(args, sem_file,dsm_file, is_HR, ref_scale=16):


    print(' [*] Reading Labels {}'.format(os.path.basename(sem_file)))

    labels = readHR(data_file=sem_file, roi_lon_lat=None, scale=args.scale, as_float=False)
    lut = np.ones(256, dtype=np.uint8) * 255
    lut[[255, 29, 179, 150, 226, 76]] = np.arange(6, dtype=np.uint8)
    labels = cv2.LUT(cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY), lut)
    # 3 vegetation
    # 5 buildings
    # 255 with or without gt

    if is_HR: ref_scale = ref_scale //args.scale

    get_median = lambda x,axis: np.percentile(x,50,axis=axis, interpolation='nearest')
    labels = block_reduce(labels,(ref_scale,ref_scale),get_median)

    mask_out = labels == 255
    labels = np.float32(labels)
    labels[mask_out] = -1.0

    labels_reg = readHR(data_file=dsm_file, roi_lon_lat=None, scale=args.scale, as_float=False)
    labels_reg = labels_reg.astype(np.float32)

    labels_reg = block_reduce(labels_reg,(ref_scale,ref_scale),np.mean)

    labels_reg[mask_out] = -1.

    im_out = np.stack((labels,labels_reg), axis=-1)

    xmin, xmax = 0, im_out.shape[1]
    ymin, ymax = 0, im_out.shape[1]
    return im_out,(xmin, ymin, xmax, ymax)
