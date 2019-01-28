import numpy as np
from osgeo import gdal
import argparse
import re

import gdal_processing as gp
import plots
from read_geoTiff import readHR, readS2

HRFILE = '/home/pf/pfstaff/projects/andresro/sparse/data/3000_gsd5.0.tif'

LRFILE = "/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml"

def parseargs():
#
    parser = argparse.ArgumentParser(description="Read Hi-Low aerial pairs", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--HR_file", help="An input sentinel-2 data file. This can be either the original ZIP file, or the S2A[...].xml file in a SAFE directory extracted from that ZIP.",
                        default = HRFILE)
    parser.add_argument("--LR_file",
                    help="An input sentinel-2 data file. This can be either the original ZIP file, or the S2A[...].xml file in a SAFE directory extracted from that ZIP.",
                    default=LRFILE)
    parser.add_argument("--roi_lon_lat", default='117.84,8.82,117.92,8.9',
        help="Sets the region of interest to extract, WGS84, decimal notation. Use this syntax: lon_1,lat_1,lon_2,lat_2. The order of points 1 and 2 does not matter: the region of interest extends to the min/max in each direction. Example: --roi_lon_lat=-1.12132,44.72408,-0.90350,44.58646")
    parser.add_argument("--select_bands", default="B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12", help="Select the bands. Using comma-separated band names.")

    args = parser.parse_args()
    return args


args = parseargs()

# LR_file_ = "/home/pf/pfstaff/projects/andresro/barry_palm/data/1C/coco_2017p/PRODUCT/S2A_MSIL1C_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL1C.xml"

LR_file = gp.getrefDataset(args.LR_file,is_use_gtiff=True)
HR_file = args.HR_file
# HR_file = '/home/pf/pfstaff/projects/andresro/sparse/data/3000/ROI5_down5.vrt'



dsH = gdal.Open(HR_file)

dsL = gdal.Open(LR_file)

Points_FILE ='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_manual.kml'


roi_lon_lat = args.roi_lon_lat


if roi_lon_lat and not roi_lon_lat == 'None':
    roi_lon1, roi_lat1, roi_lon2, roi_lat2 = map(float,re.split(',', roi_lon_lat))
else:
    roi_lon1, roi_lat1, roi_lon2, roi_lat2 = -100, -90, 180, 90


#
# lat_lon_coords = gp.read_coords(Points_FILE)
#
# hr_coords = np.zeros_like(lat_lon_coords)
# lr_coords = hr_coords.copy()
#
# for key, val in enumerate(lat_lon_coords):
#
#     xy = gp.to_xy(val[0],val[1],dsH)
#
#     hr_coords[key] = [xy[0],xy[1],0]
#
#     xy = gp.to_xy(val[0],val[1],dsL)
#
#     lr_coords[key] = [xy[0],xy[1],0]
#


xmin_H,xmax_H,ymin_H,ymax_H = gp.to_xy_box(roi_lon1,roi_lat1,roi_lon2,roi_lat2,dsH, enlarge=2)




hr_mask1 = gp.rasterize_points(Input=Points_FILE,refDataset=HR_file,lims=(xmin_H,ymin_H,xmax_H,ymax_H),scale = 2)



im = plots.plot_heatmap(hr_mask1,0,3)

im.save('output/hr_mask.png')



xmin_L,xmax_L,ymin_L,ymax_L = gp.to_xy_box(roi_lon1,roi_lat1,roi_lon2,roi_lat2,dsL)


lr_mask1 = gp.rasterize_points(Input=Points_FILE,refDataset=LR_file,lims=(xmin_L,ymin_L,xmax_L,ymax_L),scale = 10)

# lr_mask = np.zeros((y_max,x_max))
# for key,val in enumerate(lr_coords):
#     # we could already smooth labels here not by adding a 1 to each location but a gaussian to the neighbourhood.
#     x, y = int(val[0]),int(val[1])
#
#     if x >= 0 and y >= 0 and x <= x_max and y <= y_max:
#         lr_mask[y,x] +=1
#
print lr_mask1.sum()
#
im = plots.plot_heatmap(lr_mask1,0,3)

im.save('output/lr_mask.png')



dataH = readHR(args)
print 'High-res shape:'
print dataH.shape, map(lambda x: x/2.0, dataH.shape)

data10, data20 = readS2(args)
print 'Low-res shapes (10m, 20m):'
print data10.shape, data20.shape


print('done!')


