import numpy as np
from osgeo import gdal

import gdal_processing as gp
import plots


LR_file = "/home/pf/pfstaff/projects/andresro/barry_palm/data/1C/coco_2017p/PRODUCT/S2A_MSIL1C_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL1C.xml"


HR_file = '/home/pf/pfstaff/projects/andresro/sparse/data/3000/ROI1.tif'


dsH = gdal.Open(HR_file)

dsL = gdal.Open(gp.getrefDataset(LR_file))

Points_FILE ='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_manual.kml'




lat_lon_coords = gp.read_coords(Points_FILE)

hr_coords = np.zeros_like(lat_lon_coords)
lr_coords = np.zeros_like(lat_lon_coords)
for key, val in enumerate(lat_lon_coords):

    xy = gp.to_xy(val[0],val[1],dsH)

    hr_coords[key] = [xy[0],xy[1],0]

    xy = gp.to_xy(val[0],val[1],dsL)

    lr_coords[key] = [xy[0],xy[1],0]


# hr_raster = dsH.ReadAsArray()
#
# print hr_raster.shape
#
# x_max, y_max = hr_raster.shape[2], hr_raster.shape[1]
#
# hr_mask1 = gp.rasterize_points(Input=Points_FILE,refDataset=HR_file,lims=(0,0,x_max,y_max),scale = 1)
#
# hr_mask = np.zeros((x_max,y_max))
# for key,val in enumerate(hr_coords):
#     # we could already smooth labels here not by adding a 1 to each location but a gaussian to the neighbourhood.
#     x, y = val[0],val[1]
#
#     if x >= 0 and y >= 0 and x <= x_max and y <= y_max:
#         hr_mask[y,x] +=1
#
# im = plots.plot_heatmap(hr_mask,0,2)
#
# im.save('hr_maksk.png')
#



x_max, y_max = dsL.RasterXSize, dsL.RasterYSize

# lr_mask1 = gp.rasterize_points(Input=Points_FILE,refDataset=LR_file,lims=(0,0,y_max,x_max),scale = 1)

lr_mask = np.zeros((y_max,x_max))
for key,val in enumerate(lr_coords):
    # we could already smooth labels here not by adding a 1 to each location but a gaussian to the neighbourhood.
    x, y = int(val[0]),int(val[1])

    if x >= 0 and y >= 0 and x <= x_max and y <= y_max:
        lr_mask[y,x] +=1

im = plots.plot_heatmap(lr_mask,0,2)

im.save('lr_maksk.png')


print('done!')


