
# coding: utf-8
from osgeo import gdal, ogr, osr
import os, sys
import glob
import simplekml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import re

from plots import plot_heatmap
import gdal_processing as gp
from read_geoTiff import readHR
# In[1]:

# p2ha = lambda x: (x*10)**2 /100**2
#
#
# # In[2]:
#
#
#
# def loop_zonal_stats_update(input_zone_polygon, input_value_raster, fieldname, fn, is_update=True, refband=1, is_pos_only=False,bias=1, field_name = 'Name'):
#
#     shp = ogr.Open(input_zone_polygon, update=1)
#     lyr = shp.GetLayer()
#     lyrdf =lyr.GetLayerDefn()
#
#
#     id_ = lyrdf.GetFieldIndex(fieldname)
#     if id_ == -1 and is_update:
#         field_defn = ogr.FieldDefn(fieldname, ogr.OFTReal)
#         lyr.CreateField(field_defn)
#         id_ = lyrdf.GetFieldIndex(fieldname)
#     else:
#         print('Field {} already exists, may overwrite'.format(fieldname))
#     outVals = []
#     id_Name = lyrdf.GetFieldIndex(field_name)
#
#     for FID in tqdm(range(lyr.GetFeatureCount())):
#         feat = lyr.GetFeature(FID)
#         if feat is not None:
#             # compute sum
#             name_ = feat.GetField(id_Name)
#             meanValue = zonal_stats(FID, input_zone_polygon, input_value_raster, fn, refband=refband,bias=bias)
# #             print(f' {meanValue:.2f} Trees in {name_}')
#             outVals.append(meanValue)
# #             if np.isnan(meanValue):
# #                 print(name_,FID,'is all nan')
#             if is_update:
#                 lyr.SetFeature(feat)
#                 feat.SetField(id_,meanValue)
#                 lyr.SetFeature(feat)
#     return outVals
#
# def zonal_stats(FID, input_zone_polygon, input_value_raster, fn, is_return_numpoints = False, refband=1, bias = 1.0):
#
#     # Open data
#     raster = gdal.Open(input_value_raster)
#     shp = ogr.Open(input_zone_polygon)
#     lyr = shp.GetLayer()
#
#     # Get raster georeference info
#     transform = raster.GetGeoTransform()
#     xOrigin = transform[0]
#     yOrigin = transform[3]
#     pixelWidth = transform[1]
#     pixelHeight = transform[5]
#
#     # Reproject vector geometry to same projection as raster
#     sourceSR = lyr.GetSpatialRef()
#     targetSR = osr.SpatialReference()
#     targetSR.ImportFromWkt(raster.GetProjectionRef())
#     coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
#     feat = lyr.GetFeature(FID)
#     geom = feat.GetGeometryRef()
#     geom.Transform(coordTrans)
#
#     # Get extent of feat
#     geom = feat.GetGeometryRef()
#     if (geom.GetGeometryName() == 'MULTIPOLYGON'):
#         count = 0
#         pointsX = []; pointsY = []
#         for polygon in geom:
#             geomInner = geom.GetGeometryRef(count)
#             ring = geomInner.GetGeometryRef(0)
#             numpoints = ring.GetPointCount()
#             for p in range(numpoints):
#                     lon, lat, z = ring.GetPoint(p)
#                     pointsX.append(lon)
#                     pointsY.append(lat)
#             count += 1
#     elif geom.GetGeometryName() == 'POLYGON':
#         ring = geom.GetGeometryRef(0)
#         numpoints = ring.GetPointCount()
#         pointsX = []; pointsY = []
#         for p in range(numpoints):
#                 lon, lat, z = ring.GetPoint(p)
#                 pointsX.append(lon)
#                 pointsY.append(lat)
#     else:
#         sys.exit("ERROR: Geometry needs to be a Polygon")
#     xmin = min(pointsX)
#     xmax = max(pointsX)
#     ymin = min(pointsY)
#     ymax = max(pointsY)
#
#     # Specify offset and rows and columns to read
#     xoff = int((xmin - xOrigin)/pixelWidth)
#     yoff = int((yOrigin - ymax)/pixelWidth)
#
#     xcount = int((xmax - xmin)/pixelWidth)+1
#     ycount = int((ymax - ymin)/pixelWidth)+1
#
#
#     xoff = min(xoff,raster.RasterXSize -1)
#     xoff = max(xoff,1)
#
#     xcount = min(xcount,raster.RasterXSize -1 - xoff)
#     ycount = min(ycount,raster.RasterYSize -1 - yoff)
#
#
#     # Create memory target raster
#     target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
#     target_ds.SetGeoTransform((
#         xmin, pixelWidth, 0,
#         ymax, 0, pixelHeight,
#     ))
#
#     # Create for target raster the same projection as for the value raster
#     raster_srs = osr.SpatialReference()
#     raster_srs.ImportFromWkt(raster.GetProjectionRef())
#     target_ds.SetProjection(raster_srs.ExportToWkt())
#
#     # Rasterize zone polygon to raster
#     gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])
#
#     # Read raster as arrays
#     banddataraster = raster.GetRasterBand(refband)
#     try:
#         dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
#     except AttributeError:
#         print('dataraster wrong')
# #         print('geotransform',transform)
#         print(xoff,yoff,xcount,ycount)
#         print(raster.RasterXSize,raster.RasterYSize, 'xmax,ymax:',xoff+xcount,yoff+xcount)
#         return np.nan
#
#     bandmask = target_ds.GetRasterBand(1)
#     datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
# #     print(datamask.mean())
#     clip = True
#     if clip:
# #         dataraster = np.clip(dataraster,0.01,1e9)
#         dataraster[dataraster < 0.01] = np.nan
#     dataraster[dataraster == 99] = np.nan
#
#     if not np.any(datamask):
#         print('datamask empty')
#         return np.nan
#     # Mask zone of raster
# #     zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
#     dataraster[np.logical_not(datamask)] = np.nan
#     dataraster *=bias
#     # Calculate statistics of zonal raster
#     # return numpy.average(zoneraster),numpy.mean(zoneraster),numpy.median(zoneraster),numpy.std(zoneraster),numpy.var(zoneraster)
#     try:
#         return fn(dataraster)
#     except ValueError:
#         print('fix')
#         return np.nan
#


# In[4]:


# obj='palm'

# object_dict= {'palm':0,'coco':1}

# ref_band = object_dict[obj]

# points ='/home/pf/pfstud/andresro/tree_annotationsAug2019/annotations/Jan/palm/49MCV/Palm_Jan_1.kml'


# In[5]:
# In[12]:


# for automatic GT
# data_config = {'T47NQA':'101.45,0.53,101.62,0.55'}


# ## Evaluate state-wide predictions

# In[13]:


# ### Split raster into blocks and save it as shapefile

# In[5]:


# import math
# EARTH_RADIUS = 6371000  # Radius in meters of Earth
# Compute the shortest path curved distance between 2 points (lat1,lon1) and (lat2,lon2) using the Haversine formula.
# def haversine_distance(lon1, lat1, lon2, lat2):
#
#     a = math.sin(math.radians((lat2 - lat1) / 2.0)) ** 2 + math.cos(math.radians(lat1)) * math.cos(
#         math.radians(lat2)) * math.sin(math.radians((lon2 - lon1) / 2.0)) ** 2
#     c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
#     return EARTH_RADIUS * c
#
# def split_roi_to_rois(lon1_, lat1_, lon2_, lat2_, meters_split = 1500):
#
#     lon1, lat1, lon2, lat2 = min(lon1_, lon2_), min(lat1_,lat2_), max(lon1_, lon2_), max(lat1_, lat2_)
#
#     delta_lon_m = haversine_distance(lon1=lon1,lat1=lat1,lon2=lon2,lat2=lat1)
#     delta_lat_m = haversine_distance(lon1=lon1,lat1=lat1,lon2=lon1,lat2=lat2)
#     rois = []
#
#     N_lon, N_lat = map(lambda x: int(math.ceil(x / meters_split)), [delta_lon_m,delta_lat_m])
#
#     delta_lon, delta_lat = (lon2-lon1, lat2 - lat1)
#     for i in range(N_lat):
#         for j in range(N_lon):
#             ind = i * N_lon + j
#             rois.append({"roi": (
#                                 lat1 + (delta_lat) * i / N_lat,
#                                 lon1 + (delta_lon) * j / N_lon,
#                                 lat1 + (delta_lat) * (i + 1) / N_lat,
#                                 lon1 + (delta_lon) * (j + 1) / N_lon),
#                         "name": "ROI{}".format(ind + 1)})
#
#     return rois

def to_bbox(roi_lon_lat):
    if isinstance(roi_lon_lat, str):
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = map(float, re.split(',', roi_lon_lat))
    else:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = roi_lon_lat

    geo_pts_ref = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]
    return geo_pts_ref



def convert_to_shp(points, is_overwrite=False):
    if points.endswith('.kml'):
        new_points=points.replace('.kml','.shp')
        if not os.path.exists(new_points) or is_overwrite:
            srcDS = gdal.OpenEx(points)
            ds = gdal.VectorTranslate(new_points, srcDS, format='ESRI Shapefile')
            ds = None
            points = new_points
    return points 

p2ha = lambda x: (x / 10) ** 2


inference_folder = '/scratch/andresro/leon_igp/sparse/inference'

folder_name = 'palmsabah_simpleA9all'
min_tree_ha = 30
scale = 400
scale_count = 10
bias = 1.1

is_overwrite = True
is_sample = True
sample_per_bin = 20
n_bins = 30
ref_raster = f'{inference_folder}/{folder_name}/0_untiled_down{scale}.tif'
ref_raster_count = f'{inference_folder}/{folder_name}/0_untiled_down{scale_count}.tif'
fname = os.path.basename(ref_raster).replace('.tif','')
save_dir = f'{inference_folder}/{folder_name}/shp/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if is_sample:
    shp_file = f"{save_dir}/{fname}_rois_{p2ha(scale)}ha_sample{sample_per_bin}_{n_bins}.shp"
else:
    shp_file = f"{save_dir}/{fname}_rois_{p2ha(scale)}ha.shp"



if is_overwrite or not os.path.isfile(shp_file):
    raster = readHR(None,data_file=ref_raster,scale=1,as_float=False)
    raster[np.isnan(raster)] = 0
    raster *=(scale**2)
    raster *= bias
    valid_pixels = raster > min_tree_ha*scale

    if is_sample:
        Trees = raster[valid_pixels].flatten()

        bins = np.histogram_bin_edges(Trees, bins=n_bins)
        Tree_bin = pd.np.digitize(Trees, bins=bins)
        dsgeo = pd.DataFrame({'Tree':Trees,'bin':Tree_bin})
        train = []
        val = []
        for _, group in dsgeo.groupby('bin', group_keys=False):
            x = group.sample(min(len(group), sample_per_bin), replace=False, random_state=999)
            n_train = int(x.shape[0] * 0.7)
            train.extend(x.index[0:n_train])
            val.extend(x.index[n_train:])
        #     print(name,x.shape)
        dsgeo.drop(['bin'], inplace=True, axis=1)
        dsgeo['is_train_loc'] = np.array([x in train for x in dsgeo.index])
        dsgeo['is_val_loc'] = np.array([x in val for x in dsgeo.index])
        # in_sample = np.logical_or(dsgeo.is_train_loc, dsgeo.is_val_loc)

        coords_train = np.argwhere(valid_pixels)[dsgeo.is_train_loc]
        coords_val = np.argwhere(valid_pixels)[dsgeo.is_val_loc]


        # valid_coords = np.argwhere(valid_pixels)[in_sample]
        valid_coords = np.concatenate((coords_train,coords_val))

        print(f'Total {len(valid_coords)}, Train 0:{len(coords_train)}, val {len(coords_train)+1}:{len(valid_coords)}')
    else:
        valid_coords = np.argwhere(valid_pixels)


    roi1 = []
    ds = gdal.Open(ref_raster)
    for i,(y,x) in enumerate(valid_coords):
        lat1,lon1 = gp.to_latlon(x,y,ds=ds)
        lat2, lon2 = gp.to_latlon(x+1, y+1, ds=ds)
        roi1.append({'name':f'ROI{i}',
                    'roi':(lat1, lon1, lat2, lon2)})



    kmlfile_name = shp_file.replace('.shp','.kml')
    kml = simplekml.Kml()
    for roi in roi1:
        lat1, lon1, lat2, lon2 = roi["roi"]
        # print roi

        geo_pts_ref = to_bbox([lon1, lat1, lon2, lat2])
        pol = kml.newpolygon(name=roi['name'])
        pol.outerboundaryis = geo_pts_ref
        # pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.white)

    kml.save(kmlfile_name)
    shp_file = convert_to_shp(kmlfile_name, is_overwrite=True)
    print(shp_file, len(roi1))

    # Add tree count

    fieldname = 'Trees'




    def func_(x):
        x *= (scale_count ** 2)
        x *= bias
        is_palm = x > min_tree_ha * p2ha(scale_count)
        trees = np.nansum(x[is_palm])
        return trees

    from zonal_stats import loop_zonal_stats_update, save_csv

    loop_zonal_stats_update(shp_file, ref_raster_count, fieldname, fn=func_)

csv_file = save_csv(shp_file)





#
# # In[ ]:
# if len(roi1) > 5000:
#
#     splits = len(roi1)// 5000 +1
#
#     for i in range(splits):
#         start,stop = i*5000,(i+1)*5000
#         print(start,stop)
#         roi2 = roi1[start:stop]
#         fname = os.path.basename(ref_raster).replace('.tif','')
#         kmlfile_name = f"{save_dir}/{fname}_rois_{p2ha(scale)}ha_reduced{start}.kml"
#         kml = simplekml.Kml()
#         for roi in roi2:
#             lat1, lon1, lat2, lon2 = roi["roi"]
#
#             geo_pts_ref = to_bbox([lon1, lat1, lon2, lat2])
#             pol = kml.newpolygon(name=roi['name'])
#             pol.outerboundaryis = geo_pts_ref
#             # pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.white)
#
#         kml.save(kmlfile_name)
#         points = convert_to_shp(kmlfile_name, is_overwrite=True)
#         print(points)


    # In[ ]:
    #
    #
    # fname = os.path.basename(ref_raster).replace('.tif','')
    # kmlfile_name = f"{save_dir}/{fname}_rois_{len(roi_)}.kml"
    # kml = simplekml.Kml()
    # for roi in roi_:
    #     lat1, lon1, lat2, lon2 = roi["roi"]
    #     # print roi
    #
    #     geo_pts_ref = to_bbox([lon1, lat1, lon2, lat2])
    #     pol = kml.newpolygon(name=roi['name'])
    #     pol.outerboundaryis = geo_pts_ref
    #     # pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.white)
    #
    # kml.save(kmlfile_name)
    # print(kmlfile_name)
    #
    #
    # # In[154]:
    #
    #
    # points = convert_to_shp(kmlfile_name, is_overwrite=True)
    # print(points)
    #
    #
    #
    # # In[159]:
    #
    #
    # loop_zonal_stats_update(input_zone_polygon=points,input_value_raster=ref_raster,fieldname='pred_palm',fn=np.nansum, is_update=True, refband=1,bias=1.5)
    #
    #
    # # In[153]:
    #
    #
    # points
    #
    #
    # # In[147]:
    #
    #
    # kmlfile_name
    #
