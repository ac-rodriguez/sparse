import os, sys
import glob
import re
import shutil
from osgeo import ogr, gdal, osr
import numpy as np
from shapely.geometry import Polygon
from shapely import ops
from shapely.wkt import loads
from functools import partial
import pyproj
from scipy import ndimage
from skimage.measure import block_reduce

driver = ogr.GetDriverByName('ESRI Shapefile')

def reproject_layer(InputVector,inSpatialRef = None,outSpatialRef = None):


    # input SpatialReference
    source_ds = ogr.Open(InputVector)
    source_layer = source_ds.GetLayer()
    inSpatialRef = source_layer.GetSpatialRef()
    # inSpatialRef = osr.SpatialReference()
    # inSpatialRef.ImportFromEPSG(2927)

    # output SpatialReference
    outSpatialRef = osr.SpatialReference()
    outSpatialRef.ImportFromEPSG(32647)

    # create the CoordinateTransformation
    coordTrans = osr.CoordinateTransformation(inSpatialRef, outSpatialRef)

    # get the input layer
    inDataSet = ogr.Open(InputVector)
    inLayer = inDataSet.GetLayer()

    # create the output layer
    outputShapefile = 'output.shp'
    if os.path.exists(outputShapefile):
        driver.DeleteDataSource(outputShapefile)
    outDataSet = driver.CreateDataSource(outputShapefile)
    outLayer = outDataSet.CreateLayer("basemap_32647", geom_type=ogr.wkbMultiPolygon)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTrans)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # dereference the features and get the next input feature
        outFeature = None
        inFeature = inLayer.GetNextFeature()
    # Save and close the shapefiles
    # inDataSet = None
    # outDataSet = None


def get_points(Input):
    # Open Shapefile
    Shapefile = ogr.Open(Input)
    n_layers = Shapefile.GetLayerCount()
    points = []
    for j in range(n_layers):
        Shapefile_layer = Shapefile.GetLayer()

        n_points = Shapefile_layer.GetFeatureCount()

        for i in range(n_points):
            feat = Shapefile_layer.GetNextFeature()
            geom = feat.geometry()
            points.extend(geom.GetPoints())
    print(Input, len(points))
    return np.array(points)

def read_coords(Input):
    if os.path.isdir(Input):
        filename = os.path.join(Input, 'Detections.npy')
        if not os.path.isfile(filename):
            points = None
            filename_list = glob.glob(os.path.join(Input, '*.kml'))
            if len(filename_list) == 0:
                print('No .kml files were found in {}'.format(Input))
                sys.exit(1)
            for file in filename_list:
                points_ = get_points(file)
                points = np.concatenate((points, points_)) if points is not None else points_
            np.save(filename, points)
        else:
            points = np.load(filename)
            print('GT Points loaded from {}'.format(filename))
    else:
        filename = Input.replace(".kml", ".npy")

        if not os.path.isfile(filename):
            points = get_points(Input)
            np.save(filename,points)
        else:
            points = np.load(filename)
            print('GT Points loaded from {}'.format(filename))

    return points

def rasterize_points_constrained(Input, refDataset, lims, lims_with_labels, up_scale=10, sigma=None, sq_kernel=False):

    is_sq_kernel = sq_kernel is not None
    lims = [i * up_scale for i in lims]

    xmin, ymin, xmax, ymax = lims # points come already ordered

    lims_with_labels = [i * up_scale for i in lims_with_labels]

    xmin1, ymin1, xmax1, ymax1 = lims_with_labels


    Image = gdal.Open(refDataset)
    length_x = xmax - xmin + up_scale
    length_y = ymax - ymin + up_scale

    points = read_coords(Input)

    mask = np.zeros((length_y, length_x), dtype=np.float32)

    xoff, a, b, yoff, d, e = Image.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(Image.GetProjection())
    srsLatLon = osr.SpatialReference()
    srsLatLon.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(srsLatLon, srs)

    if sq_kernel:
        w = sq_kernel * 16 // 2

    a = a / up_scale
    e = e / up_scale

    for i in points:

        (xp, yp, h) = ct.TransformPoint(i[0], i[1], 0.)
        xp -= xoff
        yp -= yoff
        # matrix inversion
        det_inv = 1. / (a * e - d * b)
        x = (e * xp - b * yp) * det_inv
        y = (-d * xp + a * yp) * det_inv
        x, y = (int(x), int(y))
        # x,y = to_xy(i[0], i[1], Image)
        if x >= xmin and x <=xmax and y>=ymin and y<=ymax:
            if x >= xmin1 and x <= xmax1 and y >= ymin1 and y <= ymax1:
                x1 = x - xmin
                y1 = y - ymin
                if is_sq_kernel:
                    mask[max(0,y1-w):(y1+w),max(0,x1-w):(x1+w)] += 1/float((w*2)**2)
                else:
                    mask[y1,x1] += 1
    z_norm = np.sum(mask)
    print(' max density = {}'.format(mask.max()))

    if not is_sq_kernel:
        if sigma is None:
            sigma = up_scale / np.pi
        mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)

    print(' Total points: {}'.format(np.sum(mask[mask>-1])))


    mask = block_reduce(mask, (up_scale, up_scale), np.sum)

    threshold = 0
    if not is_sq_kernel:
        mask = mask * (z_norm / np.sum(mask))
        threshold = 1e-5

    print('GT points were computed on a {} times larger area than RefData'.format(up_scale))

    if is_sq_kernel:
        print('smoothed with a Squared Kernel of size = {:.2f} and downsampled x{} to original res'.format(w,up_scale))
    else:
        print('smoothed with a Gaussian \sigma = {:.2f} and downsampled x{} to original res'.format(sigma,up_scale))

    print(' max density = {} (after smoothing)'.format(mask.max()))
    scale_f = float(up_scale)
    # Set points outside of constraint to -1
    mask_bool = np.zeros_like(mask)
    mask_bool[:,:max(0,int((xmin1-xmin) / scale_f))] = -1.
    mask_bool[:,int(np.ceil((xmax1-xmin+1) / scale_f)):] = -1.
    mask_bool[:max(0,int((ymin1-ymin) / scale_f)),:] = -1.
    mask_bool[int(np.ceil((ymax1-ymin+1) / scale_f)):,:] = -1.

    mask = np.where(mask > threshold, mask,mask_bool)
    z = float(np.sum(mask>-1))
    print(' Total points: {} (after smoothing and downscaling)'.format(np.sum(mask[mask>-1])))
    print(' Density Distribution: p'+str([0,1,25,5,75,99,100]))
    print(np.percentile(mask[mask>0],q=(0,1,25,5,75,99,100)))
    print(' Class Distribution: \n\t1:{:.4f} \n\t0:{:.4f} '.format(np.sum(mask>0)/z, np.sum(mask==0)/z))

    print('\n Distribution on LR space:')
    mask_ = block_reduce(mask, (16//up_scale, 16//up_scale), np.sum)
    z = float(np.sum(mask_ > -1))
    print(' Total points: {}'.format(np.sum(mask_[mask_ > -1])))
    print(' Density Distribution: p' + str([0, 1, 25, 5, 75, 99, 100]))
    print(np.percentile(mask_[mask_ > 0], q=(0, 1, 25, 5, 75, 99, 100)))
    print(' Class Distribution: \n\t1:{:.4f} \n\t0:{:.4f} '.format(np.sum(mask_ > 0) / z, np.sum(mask_ == 0) / z))


    print(' Masked pixels: {} / {} ({:.2f}%)'.format(np.sum(mask == -1),mask.shape[0]*mask.shape[1],100.* np.sum(mask == -1) /float(mask.shape[0]*mask.shape[1])))
    print('Image size: width={} x height={}'.format(mask.shape[1],mask.shape[0]))
    return mask

def rasterize_points(Input, refDataset, lims, scale = 10):

    lims = [i*scale for i in lims]

    xmin, ymin, xmax, ymax = lims
    Image = gdal.Open(refDataset)
    length_x = xmax - xmin + 1
    length_y = ymax - ymin + 1

    points = read_coords(Input)

    mask = np.zeros((length_y, length_x), dtype=np.int32)

    xoff, a, b, yoff, d, e = Image.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(Image.GetProjection())
    srsLatLon = osr.SpatialReference()
    srsLatLon.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(srsLatLon, srs)


    a = a / scale
    e = e / scale

    for i in points:

        (xp, yp, h) = ct.TransformPoint(i[0], i[1], 0.)
        xp -= xoff
        yp -= yoff
        # matrix inversion
        det_inv = 1. / (a * e - d * b)
        x = (e * xp - b * yp) * det_inv
        y = (-d * xp + a * yp) * det_inv
        x, y = (int(x), int(y))
        # x,y = to_xy(i[0], i[1], Image)
        if x >= xmin and x <=xmax and y>=ymin and y<=ymax:
            x1 = x - xmin
            y1 = y - ymin

            mask[y1,x1]+=1

    if scale > 1:
        # sigma = scale
        sigma = scale / np.pi
        mask = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma)

        mask = block_reduce(mask, (scale, scale), np.sum)
        # mask = mask[::scale,::scale]
        print('GT points were smoothed on High resolution with a Gaussian \sigma = {:.2f} and downsampled {} times'.format(sigma, scale))

    return mask
def rasterize_points_tiff(Input, refDataset, overwrite = False):

    Image = gdal.Open(refDataset)

    if not '.xml' in Input:
        OutputImage = os.path.join(Input, 'Detections.tif')
    else:
        OutputImage = Input.replace('.xml', '.tif')
    if not os.path.exists(OutputImage) or overwrite:
        options = ['alpha=yes']
        target_ds = gdal.GetDriverByName('GTiff').Create(OutputImage, Image.RasterXSize, Image.RasterYSize, 4, gdal.GDT_Byte, options=options)

        target_ds.SetGeoTransform(Image.GetGeoTransform())
        target_ds.SetProjection(Image.GetProjectionRef())


        band = target_ds.GetRasterBand(1)
        band.SetNoDataValue(-1)

        filename_list = glob.glob(os.path.join(Input, '*.kml'))
        if len(filename_list) == 0:
            print('No .kml files were found in {}'.format(Input))
            sys.exit(1)

        for InputVector in filename_list:
            Shapefile = ogr.Open(InputVector)
            Shapefile_layer = Shapefile.GetLayer()
            # Rasterize
            gdal.RasterizeLayer(target_ds, [1,2,3,4], Shapefile_layer, burn_values=[255, 0, 0, 100])

        print('{} saved!'.format(OutputImage))
    else:
        print('{} already exists!'.format(OutputImage))

def getrefDataset(refds,is_use_gtiff =False):

    raster1c = gdal.Open(refds)
    datasets1c = raster1c.GetSubDatasets()

    if not datasets1c and is_use_gtiff:
        parentfile = os.path.dirname(refds)
        geotiffile = os.path.join(parentfile,'geotif','Band_B3.tif')
        if os.path.isfile(geotiffile):
            return geotiffile

    tenMsets = []
    for (dsname, dsdesc) in datasets1c:
        if '10m resolution' in dsdesc:
            tenMsets += [(dsname, dsdesc)]
    return  tenMsets[0][0]

def get_jp2(path, band, res=10):
    if path.endswith('.xml'):
        path = os.path.dirname(path)
    if band == 'CLD':
        cld = glob.glob(f"{path}/GRANULE/*/QI_DATA/*_CLD_{res}m.jp2") + glob.glob(f"{path}/GRANULE/*/QI_DATA/*_CLDPRB_{res}m.jp2")
        return cld[0]
    else:
        return glob.glob(f"{path}/GRANULE/*/IMG_DATA/R{res}m/*_{band}_{res}m.jp2")[0]


def rasterize_numpy(Input, refDataset, filename='ProjectedNumpy.tif', type=gdal.GDT_Byte, roi_lon_lat = None):
    if type == 'float32':
        type = gdal.GDT_Float32
    Image = gdal.Open(refDataset)
    if roi_lon_lat:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = split_roi_string(roi_lon_lat)
        xmin, ymin, xmax, ymax = to_xy_box((roi_lon1, roi_lat1, roi_lon2, roi_lat2), Image, enlarge=1)
    else:
        xmin, ymin = 0, 0
        xmax, ymax = Image.RasterXSize - 1, Image.RasterYSize - 1

    if len(Input.shape) == 2:
        Input = np.expand_dims(Input, axis = 2)

    nbands = Input.shape[-1]

    options = ['alpha=yes']
    target_ds = gdal.GetDriverByName('GTiff').Create(filename, xmax - xmin + 1, ymax - ymin + 1, nbands, type, options=options)

    ox, pw, a, oy,b, ph = Image.GetGeoTransform()
    ox = xmin *pw + ox
    oy = ymin *ph + oy
    target_ds.SetGeoTransform((ox, pw, a, oy,b, ph))
    target_ds.SetProjection(Image.GetProjectionRef())

    for i in range(nbands):
        band = target_ds.GetRasterBand(i+1)
        band.WriteArray(Input[...,i], 0, 0)
    target_ds.FlushCache()
    target_ds = None

    print('{} saved!'.format(filename))


def rasterize_polygons(InputVector, refDataset, lims=None, offset=None, attribute=None, NoDataValue=0):
    Image = gdal.Open(refDataset)

    burnVal = 1  # value for the output image pixels

    # Open Shapefile
    Shapefile = ogr.Open(InputVector)
    Shapefile_layer = Shapefile.GetLayer()

    ### Rasterise
    # Output = gdal.GetDriverByName(GTiff).Create('temp-palm.tif', Image.RasterXSize, Image.RasterYSize, 1, gdal.GDT_Byte)
    if attribute is None:
        Output = gdal.GetDriverByName('MEM').Create("", Image.RasterXSize, Image.RasterYSize, 1, gdal.GDT_Int32)
    else:
        Output = gdal.GetDriverByName('MEM').Create("", Image.RasterXSize, Image.RasterYSize, 1, gdal.GDT_Float32)

    Output.SetGeoTransform(Image.GetGeoTransform())
    Output.SetProjection(Image.GetProjectionRef())

    # Write data to band 1
    Band = Output.GetRasterBand(1)
    Band.SetNoDataValue(NoDataValue)
    if attribute is None:
        gdal.RasterizeLayer(Output, [1], Shapefile_layer, burn_values=[burnVal])
    else:
        gdal.RasterizeLayer(Output, [1], Shapefile_layer, options=[f'ATTRIBUTE={attribute}'])


    if offset is not None:
        mask = Output.ReadAsArray(*offset)
    elif lims is None:
        mask = Output.ReadAsArray()
    else:
        xmin, ymin, xmax, ymax = lims
        mask = Output.ReadAsArray(xoff=xmin, yoff=ymin, xsize=xmax - xmin + 1, ysize=ymax - ymin + 1,
                               buf_xsize=xmax - xmin + 1,
                               buf_ysize=ymax - ymin + 1)
    if mask is None:
        return None
    if mask.max() == 0:
        print(f' [!] Empty mask in the ROI {InputVector}...')

    return mask

def to_xy(lon, lat, ds, is_int = True):

    xoff, a, b, yoff, d, e = ds.GetGeoTransform()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(ds.GetProjection())
    srsLatLon = osr.SpatialReference()
    srsLatLon.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(srsLatLon, srs)

    (xp, yp, h) = ct.TransformPoint(lon, lat, 0.)
    xp -= xoff
    yp -= yoff
    # matrix inversion
    det_inv = 1. / (a * e - d * b)
    x = (e * xp - b * yp) * det_inv
    y = (-d * xp + a * yp) * det_inv
    if is_int:
        return (int(x), int(y))
    else:
        return (x,y)


def split_roi_string(roi_lon_lat):
    if isinstance(roi_lon_lat,str):
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = [float(x) for x in re.split(',', roi_lon_lat)]
    else:
        roi_lon1, roi_lat1, roi_lon2, roi_lat2 = roi_lon_lat
    return max(roi_lon1,roi_lon2), max(roi_lat1,roi_lat2), min(roi_lon1,roi_lon2), min(roi_lat1,roi_lat2)

def to_xy_box(lims,dsREF, enlarge = 1):
    enlarge = float(enlarge)
    roi_lon1, roi_lat1, roi_lon2, roi_lat2 = split_roi_string(lims)

    roi_intersection(dsREF,geo_pts_ref= [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)])

    x1, y1 = to_xy(roi_lon1, roi_lat1, dsREF)
    x2, y2 = to_xy(roi_lon2, roi_lat2, dsREF)
    xmin = max(min(x1, x2, dsREF.RasterXSize - 1), 0)
    xmax = min(max(x1, x2, 0), dsREF.RasterXSize - 1)
    ymin = max(min(y1, y2, dsREF.RasterYSize - 1), 0)
    ymax = min(max(y1, y2, 0), dsREF.RasterYSize - 1)

    of_x = xmax - xmin + 1
    of_y = ymax - ymin + 1

    of_x = np.int(of_x / enlarge) * enlarge
    of_y = np.int(of_y / enlarge) * enlarge
    # xmax = int((xmax + 1) / enlarge) * enlarge - 1
    #
    # xmin = int(xmin / enlarge) * enlarge
    # xmax = int((xmax + 1) / enlarge) * enlarge - 1
    #
    # ymin = int(ymin / enlarge) * enlarge
    # ymax = int((ymax + 1) / enlarge) * enlarge - 1

    bbox = (xmin, ymin, xmin+of_x-1, ymin+of_y-1)
    return [int(x) for x in bbox]


def getGeom(inputfile, shapely = False):
    Shapefile = ogr.Open(inputfile)
    n_layers = Shapefile.GetLayerCount()
    wkt_list  = []
    for _ in range(n_layers):
        Shapefile_layer = Shapefile.GetLayer()

        n_points = Shapefile_layer.GetFeatureCount()

        for _ in range(n_points):
            feat = Shapefile_layer.GetNextFeature()
            if feat:
                geom = feat.geometry().ExportToWkt()
                if shapely:
                    geom = loads(geom)
                wkt_list.append(geom)

    print('{} geometries loaded from {}'.format(len(wkt_list),inputfile))

    return wkt_list


def split_shapefile(filename,dst_path):

    driver = ogr.GetDriverByName('KML')
    dataSource = ogr.Open(filename)
    layer = dataSource.GetLayer()
    sr = layer.GetSpatialRef()  # Spatial Reference

    # dst = path + "/kml"  # Output directory
    if not os.path.isdir(dst_path):     os.mkdir(dst_path)
    new_feat = ogr.Feature(layer.GetLayerDefn())  # Dummy feature

    for id, feat in enumerate(layer):
        filename_ = os.path.join(dst_path, '{}_{}_{}.kml'.format(feat.GetField(6),feat.GetField(8), feat.GetField(7)))

        if not os.path.isfile(filename_):
            new_ds = driver.CreateDataSource(filename_)
            new_lyr = new_ds.CreateLayer('feat_{}'.format(id), sr, ogr.wkbPolygon)  # You have to specify the geometry type the layer will contain here with an ogr constant. I assume it is polygon but it can be changed.
            geom = feat.geometry().Clone()
            new_feat.SetGeometry(geom)
            new_lyr.CreateFeature(new_feat)

            del new_ds, new_lyr

def bbox_to_offset(bbox, ds):
    '''
    :param bbox: bbox (Extent) of a Dataset in lon-lat coordinates
    :param ds: Reference dataset to obtain the offset coordinates
    :return: Offset to be used in ReadAsArray()
    '''
    x, y = to_xy(bbox[0], bbox[2], ds)
    x1, y1 = to_xy(bbox[1], bbox[3], ds)
    xmin, xmax, ymin, ymax = min(x, x1), max(x, x1), min(y, y1), max(y, y1)

    xmin, xmax = map(lambda x: np.clip(x, 0, ds.RasterXSize), [xmin, xmax])
    ymin, ymax = map(lambda x: np.clip(x, 0, ds.RasterYSize), [ymin, ymax])

    return xmin, ymin, (xmax - xmin), (ymax - ymin)



#TODO
def to_latlon(x, y, ds):
    bag_gtrn = ds.GetGeoTransform()
    bag_proj = ds.GetProjectionRef()
    bag_srs = osr.SpatialReference(bag_proj)
    geo_srs = bag_srs.CloneGeogCS()
    transform = osr.CoordinateTransformation(bag_srs, geo_srs)

    # in a north up image:
    originX = bag_gtrn[0]
    originY = bag_gtrn[3]
    pixelWidth = bag_gtrn[1]
    pixelHeight = bag_gtrn[5]


    easting = originX + pixelWidth * x + bag_gtrn[2] * y
    northing = originY + bag_gtrn[4] * x + pixelHeight * y

    geo_pt = transform.TransformPoint(easting, northing)[:2]
    lon = geo_pt[0]
    lat = geo_pt[1]
    return lat, lon


def get_lonlat(ds, verbose= False):

    bag_gtrn = ds.GetGeoTransform()
    bag_proj = ds.GetProjectionRef()
    bag_srs = osr.SpatialReference(bag_proj)
    geo_srs = bag_srs.CloneGeogCS()
    transform = osr.CoordinateTransformation(bag_srs, geo_srs)

    bag_bbox_cells = (
        (0., 0.),
        (0, ds.RasterYSize),
        (ds.RasterXSize, ds.RasterYSize),
        (ds.RasterXSize, 0),
    )

    geo_pts = []
    for x, y in bag_bbox_cells:
        x2 = bag_gtrn[0] + bag_gtrn[1] * x + bag_gtrn[2] * y
        y2 = bag_gtrn[3] + bag_gtrn[4] * x + bag_gtrn[5] * y
        geo_pt = transform.TransformPoint(x2, y2)[:2]
        geo_pts.append(geo_pt)
        if verbose:
            print(x, y, '->', geo_pt)

    return geo_pts


def roi_intersection(ds, geo_pts_ref, return_polygon = False):
    geo_pts = get_lonlat(ds)

    p1 = Polygon(geo_pts)
    p2 = Polygon(geo_pts_ref)
    intersection = p1.intersection(p2)
    # print(intersection.area)

    if p1.intersects(p2):
        geom_area = ops.transform(
            partial(
                pyproj.transform,
                pyproj.Proj(init='EPSG:4326'),
                pyproj.Proj(
                    proj='aea',
                    lat_1=intersection.bounds[1],
                    lat_2=intersection.bounds[3])),
            intersection)

        # Print the area in m^2
        print('ROI intersection area {:.1f} Ha '.format(geom_area.area/(100.**2)))
    if return_polygon:
        return p1.intersects(p2), geom_area
    else:
        return p1.intersects(p2), None


def enlarge_pixel(xmin, xmax, ref=6):
    xmin = int(xmin / ref) * ref
    xmax = int((xmax + 1) / ref) * ref - 1
    return xmin, xmax


def pad(array, reference, offset):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros(reference.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offset[dim], offset[dim] + array.shape[dim]) for dim in range(array.ndim)]
    # Insert the array in the result at the specified offsets
    result[insertHere] = array
    return result


def read_gt(refDataset, lims, ref_obj, scale_points=1,  path = None):

    rasterize_points_tiff(path,refDataset)
    points = rasterize_points(Input=path,
                                       refDataset=refDataset, lims=lims, scale=scale_points)
    print('Count Objects= {:.2f}K'.format(np.sum(points)/1e3))


    neg_obj = rasterize_polygons(
        InputVector='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/{}/neg_areas.kml'.format(ref_obj),
        refDataset=refDataset, lims=lims)


    labels = np.zeros_like(neg_obj)

    # labels[pos_obj == 1] = 1
    labels[points > 0.5] = 1

    # Removing false positives
    labels[neg_obj == 1] = 0
    points[neg_obj == 1] = 0

    if not np.all(points == 0):
        print('Count Objects (after manually removing FP)= {:.2f}K \n Objects per pixel = {:.3f} Percentiles(1={:.3f}, 99={:.3f})'.format(
            np.sum(points) / 1e3, np.sum(points) / (points.shape[0]*points.shape[1]), np.percentile(points,1), np.percentile(points,99)))

        print('Unconditioned counts: SUM Average, P50, P99')
        print('{} && {} & {} & {} '.format(
            np.sum(points) , np.sum(points) / (points.shape[0]*points.shape[1]), np.percentile(points,50), np.percentile(points,99)))

        points1 = points.flatten()
        points1 = points1[points1 > 0.5]
        print('Conditioned counts (points > 0.5): SUM Average, P1, P99')
        print('{} && {} & {} & {} '.format(
            np.sum(points1) , np.sum(points1) / (np.sum(labels == 1)), np.percentile(points1,1), np.percentile(points1,99)))
    else:
        print('All patches in Dataset are negative examples')
    # sys.exit(0)
    return labels, points

# def smooth_and_downscale(data, scale, sigma = None, func = np.sum):
#
#     if sigma is None:
#         sigma = scale / np.pi
#     data = ndimage.gaussian_filter(data.astype(np.float32), sigma=sigma)
#     return block_reduce(data, (scale, scale, 1), func)
