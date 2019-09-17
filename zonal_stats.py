import gdal, ogr, osr
import numpy as np
import sys
import csv

from shapely.wkt import loads, dumps
import simplekml


def zonal_stats(FID, input_zone_polygon, input_value_raster, fn, is_return_numpoints = False):

    # Open data
    raster = gdal.Open(input_value_raster)
    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()

    # Get raster georeference info
    transform = raster.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # Reproject vector geometry to same projection as raster
    sourceSR = lyr.GetSpatialRef()
    targetSR = osr.SpatialReference()
    targetSR.ImportFromWkt(raster.GetProjectionRef())
    coordTrans = osr.CoordinateTransformation(sourceSR,targetSR)
    feat = lyr.GetFeature(FID)
    geom = feat.GetGeometryRef()
    geom.Transform(coordTrans)

    # Get extent of feat
    geom = feat.GetGeometryRef()

    if geom.GetGeometryName() == 'MULTIPOLYGON' :
        count = 0
        pointsX = []; pointsY = []
        for polygon in geom:
            geomInner = geom.GetGeometryRef(count)
            ring = geomInner.GetGeometryRef(0)
            numpoints = ring.GetPointCount()
            for p in range(numpoints):
                    lon, lat, z = ring.GetPoint(p)
                    pointsX.append(lon)
                    pointsY.append(lat)
            count += 1
    elif geom.GetGeometryName() == 'POLYGON':
        ring = geom.GetGeometryRef(0)
        numpoints = ring.GetPointCount()
        pointsX = []; pointsY = []
        for p in range(numpoints):
                lon, lat, z = ring.GetPoint(p)
                pointsX.append(lon)
                pointsY.append(lat)
    elif (geom.GetGeometryName() == 'LINESTRING'):
        numpoints = geom.GetPointCount()
        pointsX = []
        pointsY = []
        for p in range(numpoints):
            lon, lat, z = geom.GetPoint(p)
            pointsX.append(lon)
            pointsY.append(lat)
    else:
        sys.exit("ERROR: Geometry needs to be either Polygon or Multipolygon")

    xmin = min(pointsX)
    xmax = max(pointsX)
    ymin = min(pointsY)
    ymax = max(pointsY)

    # Specify offset and rows and columns to read
    xoff = int((xmin - xOrigin)/pixelWidth)
    yoff = int((yOrigin - ymax)/pixelWidth)
    if xoff < 0 or yoff < 0:
        return np.nan
    xcount = int((xmax - xmin)/pixelWidth)+1
    ycount = int((ymax - ymin)/pixelWidth)+1

    if is_return_numpoints:
        # TODO check that all the points are inside the region of interest
        return geom.GetPointCount()

    # Create memory target raster
    target_ds = gdal.GetDriverByName('MEM').Create('', xcount, ycount, 1, gdal.GDT_Byte)
    target_ds.SetGeoTransform((
        xmin, pixelWidth, 0,
        ymax, 0, pixelHeight,
    ))

    # Create for target raster the same projection as for the value raster
    raster_srs = osr.SpatialReference()
    raster_srs.ImportFromWkt(raster.GetProjectionRef())
    target_ds.SetProjection(raster_srs.ExportToWkt())

    # Rasterize zone polygon to raster
    gdal.RasterizeLayer(target_ds, [1], lyr, burn_values=[1])

    # Read raster as arrays
    banddataraster = raster.GetRasterBand(1)
    try:
        dataraster = banddataraster.ReadAsArray(xoff, yoff, xcount, ycount).astype(np.float)
    except AttributeError:
        return np.nan
    bandmask = target_ds.GetRasterBand(1)
    datamask = bandmask.ReadAsArray(0, 0, xcount, ycount).astype(np.float)
    clip = False
    if clip:
        dataraster = np.clip(dataraster,0.01,1e9)
    # Mask zone of raster
    # zoneraster = np.ma.masked_array(dataraster,  np.logical_not(datamask))
    zoneraster = dataraster.copy()
    zoneraster[np.logical_not(datamask)] = np.nan

    if not np.any(datamask):
        return np.nan
    # Calculate statistics of zonal raster
    # return numpy.average(zoneraster),numpy.mean(zoneraster),numpy.median(zoneraster),numpy.std(zoneraster),numpy.var(zoneraster)
    try:
        return fn(zoneraster)
    except ValueError:
        print('fix')
        return np.nan

def loop_zonal_stats(input_zone_polygon, input_value_raster,fn):

    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    # featList = range(lyr.GetFeatureCount())
    statDict = {}

    for FID in range(lyr.GetFeatureCount()):
        feat = lyr.GetFeature(FID)
        if feat is not None:
            meanValue = zonal_stats(FID, input_zone_polygon, input_value_raster, fn=fn)

            statDict[FID] = meanValue
    return statDict


def loop_zonal_stats_kml(input_zone_polygon, input_value_raster):

    shp = ogr.Open(input_zone_polygon)
    lyr = shp.GetLayer()
    # featList = range(lyr.GetFeatureCount())
    # statDict = {}
    kml = simplekml.Kml()

    for FID in range(lyr.GetFeatureCount()):
        feat = lyr.GetFeature(FID)
        if feat is not None:
            # compute sum
            meanValue = zonal_stats(FID, input_zone_polygon, input_value_raster)
            # statDict[FID] = meanValue

            # save it to kml feat
            geom = feat.GetGeometryRef()
            wkt_ = geom.ExportToWkt()

            geom = loads(wkt_)
            if geom.type != 'MultiPolygon':
                xy = geom.exterior.coords.xy
                boundaries = zip(*xy)

                pol = kml.newpolygon(name=str(FID))
                # a = val['Boundaries']
                # pol.outerboundaryis = a[0] if isinstance(a, list) and len(a) == 1 else a
                pol.outerboundaryis = boundaries
                pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.white)

                # pol.description = "<![CDATA[Properties<br>{}]]>".format(add_table(val))
                pol.description = '{:.2f}'.format(meanValue)

    filename_all = '/scratch/Dropbox/Dropbox/0_phd/yield/scofin/SOGB'
    kml.save(filename_all + "/Treecounts.kml")
    print('[*] kml saved!')

def loop_zonal_stats_update(input_zone_polygon, input_value_raster, fieldname, fn, is_update=True):

    shp = ogr.Open(input_zone_polygon, update=1)
    lyr = shp.GetLayer()
    lyrdf =lyr.GetLayerDefn()

    # TreeFieldName = 'TreePredAd1'
    if is_update:
        id_ = lyrdf.GetFieldIndex(fieldname)
        if id_ == -1:
            field_defn = ogr.FieldDefn(fieldname, ogr.OFTReal)
            lyr.CreateField(field_defn)
            id_ = lyrdf.GetFieldIndex(fieldname)
        else:
            print('Field {} already exists, may overwrite'.format(fieldname))

    id_Name = lyrdf.GetFieldIndex('Name')
    for FID in range(lyr.GetFeatureCount()):
        feat = lyr.GetFeature(FID)
        if feat is not None:
            # compute sum
            name_ = feat.GetField(id_Name)
            if 'pos' in name_:
                meanValue = zonal_stats(FID, input_zone_polygon, input_value_raster, fn)
                print(f' {meanValue:.2f} Trees in positive area')

            else:
                meanValue = zonal_stats(FID, input_zone_polygon, input_value_raster, fn, is_return_numpoints=False)
                print(f' {meanValue:.2f} Ref points')
            if np.isnan(meanValue):
                print(meanValue,FID)
            if is_update:
                lyr.SetFeature(feat)
                feat.SetField(id_,meanValue)
                lyr.SetFeature(feat)


def save_csv(shpfile,csvfile = None, with_geom = False):

    if csvfile is None:
        if shpfile.endswith('.shp'):
            csvfile = shpfile.replace('.shp','.csv')
        else:
            csvfile = shpfile.replace('.kml', '.csv')

    # Open files
    csvfile = open(csvfile, 'w')
    ds = ogr.Open(shpfile)
    lyr = ds.GetLayer()

    # Get field names
    dfn = lyr.GetLayerDefn()
    nfields = dfn.GetFieldCount()
    fields = []
    for i in range(nfields):
        fields.append(dfn.GetFieldDefn(i).GetName())
    if with_geom:
        fields.append('kmlgeometry')
    csvwriter = csv.DictWriter(csvfile, fields)
    try:
        csvwriter.writeheader()  # python 2.7+
    except:
        csvfile.write(','.join(fields) + '\n')

    # Write attributes and kml out to csv
    for feat in lyr:
        attributes = feat.items()

        if with_geom:
            geom = feat.GetGeometryRef()
            attributes['kmlgeometry'] = geom.ExportToKML()
        csvwriter.writerow(attributes)
    # clean up
    del csvwriter, lyr, ds
    csvfile.close()
    return csvfile

is_kml= False

# def main(input_zone_polygon, input_value_raster, fieldname, function=np.nanmean):
#     if is_kml:
#         loop_zonal_stats_kml(input_zone_polygon, input_value_raster, function)
#     else:
#         loop_zonal_stats_update(input_zone_polygon, input_value_raster, fieldname=fieldname, fn=function)
#
#         # return loop_zonal_stats(input_zone_polygon, input_value_raster)


if __name__ == "__main__":

    fieldname = 'Trees'
    rasterfile = '/scratch/andresro/leon_igp/sparse/inference/palmsabah_simpleA9all/0_untiled_down10.tif'
    polfile = '/scratch/andresro/leon_igp/sparse/inference/palmsabah_simpleA9all/shp/0_untiled_down400_rois_1600.0ha_reduced.shp'
    # scale = 10
    # bias = 1.1
    # min_tree_ha = 50
    # p2ha = lambda x: (x / 10) ** 2
    #
    #
    # def func_(x):
    #     x *= (scale ** 2)
    #     x *= bias
    #     is_palm = x > min_tree_ha * p2ha(scale)
    #     trees = np.nansum(x[is_palm])
    #     return trees
    #
    #
    # loop_zonal_stats_update(polfile,rasterfile,fieldname, fn=func_)

    # polfile = '/scratch/Dropbox/Dropbox/0_phd/yield/scofin/SOGB/Blocs_palmier240519_region.shp'
    save_csv(shpfile=polfile)
    print('Done!')

    # Returns for each feature a dictionary item (FID) with the statistical values in the following order: Average, Mean, Medain, Standard Deviation, Variance
    #
    # example run : $ python grid.py <full-path><output-shapefile-name>.shp xmin xmax ymin ymax gridHeight gridWidth
    #

    # if len( sys.argv ) != 3:
    #     print "[ ERROR ] you must supply two arguments: input-zone-shapefile-name.shp input-value-raster-name.tif "
    #     sys.exit( 1 )
    # print 'Returns for each feature a dictionary item (FID) with the statistical values in the following order: Average, Mean, Medain, Standard Deviation, Variance'
    # print main( sys.argv[1], sys.argv[2] )

    # # Original
    # rasterfile = '/home/pf/pfstaff/projects/andresro/barry_palm/inference/palm/Simple_May27/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_only.tif'
    # fieldname = 'TreePred'
    #
    # # Adapted
    # rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palm_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    # fieldname = 'TreePredAd'


    #SOCB dataset
    # Original
    # rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb_simple/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    # fieldname = 'TreePred1'

    # # Adapted
    # rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    # fieldname = 'PredAd1'
    #
    # rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb_count/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    # fieldname = 'count'
    # for dataset in ['other']:
    # # for dataset in ['palmsocb{}_simple_adapted'.format(x+1) for x in range(4)]:
    #     print(dataset)
    #     if dataset == 'age_simpleA':
    #         rasterfile ='/home/pf/pfstaff/projects/andresro/sparse/inference/age_simpleA/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 'age1'
    #     elif dataset == 'age4_simpleA':
    #         rasterfile ='/home/pf/pfstaff/projects/andresro/sparse/inference/age4_simpleA/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 'age2'
    #     elif dataset == 'palm_count':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palm_count/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 'pc'
    #     elif dataset == 'palm_simple':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palm_simple/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 'ps'
    #     elif dataset == 'palm_count_adapted':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palm_count_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 'pca'
    #     elif dataset == 'palm_simple_adapted':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palm_simple_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 'psa'
    #
    #     elif dataset == 'palmsocb1_simple_adapted':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb1_simple_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 's1sa'
    #         # func_ = np.ma.median
    #     elif dataset == 'palmsocb2_simple_adapted':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb2_simple_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 's2sa'
    #         # func_ = np.ma.median
    #     elif dataset == 'palmsocb3_simple_adapted':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb3_simple_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 's3sa'
    #         # func_ = np.ma.median
    #     elif dataset == 'palmsocb4_simple_adapted':
    #         rasterfile = '/home/pf/pfstaff/projects/andresro/sparse/inference/palmsocb4_simple_adapted/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE/preds_reg.tif'
    #         fieldname = 's4sa'
    #         # func_ = np.ma.median
    #     elif dataset == 'other':
    #         rasterfile = '/scratch/andresro/leon_igp/sparse/inference/palmsabah_simpleA9all/0_untiled_down10.tif'
    #         fieldname = 'Trees'
    #         # func_ = np.ma.sum
    #     else:
    #         raise ValueError('dataset {} not known'.format(dataset))

        #
        # # polfile = '/scratch/Dropbox/Dropbox/0_phd/yield/scofin/SOGB/Blocs_281118.TAB'
        # polfile = '/scratch/Dropbox/Dropbox/0_phd/yield/scofin/SOGB/Blocs_palmier240519_region.shp'
        # polfile = '/scratch/Dropbox/Dropbox/0_phd/yield/scofin/SOGB/palmier_260619_region.shp'
        # for id_ in range(3):
        #     polfile =f'/home/pf/pfstud/andresro/tree_annotationsAug2019/annotations/Jan/palm/49MCV/Palm_Jan_{id_+1}.shp'
        #
        #     print(polfile)
        #
        #     for func_ in [np.ma.sum]:
        #
        # if func_ == np.ma.median:
        #     fieldname = fieldname+'_m'






