import socket
import sys, os
import argparse
import simplekml
import shapely
from functools import partial
import pyproj

import gdal_processing as gp
parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--dataset", default='palm')
# class Bunch(object):
#   def __init__(self, adict):
#     self.__dict__.update(adict)

def get_dataset(DATASET, is_mounted = False):

    dset_config = {}
    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    if "coco" in DATASET:
        OBJECT='coco'
        dset_config['LR_file']=PATH+'/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml'
        dset_config['points']=PATH+'/barry_palm/data/labels/coco/points_detections.kml'
        dset_config['roi_lon_lat_tr'] = '117.86,8.82,117.92,8.9'
        dset_config['roi_lon_lat_val'] = '117.84,8.82,117.86,8.88'
        dset_config['roi_lon_lat_test'] = '117.81,8.82,117.84,8.88'

        # if DATASET == "coco0.3": # 2K
        #     dset_config['roi_lon_lat_tr_lb']='117.885,8.867,117.89,8.874'

        if DATASET == "coco2a": # 8.5k
            dset_config['roi_lon_lat_tr_lb']='117.88,8.88,117.891,8.895'
        elif DATASET == "coco1a":  # 8.5k
            dset_config['roi_lon_lat_tr_lb'] = '117.8821,8.8854,117.891,8.895'

        elif DATASET == "coco2b": # 8.5k
            dset_config['roi_lon_lat_tr_lb']='117.88,8.83,117.891,8.845'
        elif DATASET == "coco1b":  # 4.8k
            dset_config['roi_lon_lat_tr_lb'] = '117.8821,8.8354,117.891,8.845'
        elif DATASET == "coco2c": # 8.5k
            dset_config['roi_lon_lat_tr_lb']='117.88,8.86,117.891,8.875'
        elif DATASET == "coco1c":  # 4.8k
            dset_config['roi_lon_lat_tr_lb'] = '117.8821,8.8654,117.891,8.875'
        # elif DATASET == "coco7": # 24k
        #     dset_config['roi_lon_lat_tr_lb'] = '117.87,8.86,117.9,8.88'
        elif DATASET == "coco10": # 46k
            dset_config['roi_lon_lat_tr_lb'] = '117.86,8.85,117.9,8.88'
        elif DATASET == "coco50": # 100k
            dset_config['roi_lon_lat_tr_lb'] = '117.86,8.85,117.92,8.88'
        elif DATASET == "coco100": # 267k
            dset_config['roi_lon_lat_tr_lb'] = '117.86,8.82,117.92,8.9'

    elif "palm" in DATASET:
        OBJECT='palm'
        dset_config['LR_file']=PATH+'/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml'
        dset_config['points']=PATH+'/barry_palm/data/labels/palm/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '101.45,0.48,101.62,0.52'
        dset_config['roi_lon_lat_val'] = '101.45,0.52,101.62,0.53'
        dset_config['roi_lon_lat_test'] = '101.45,0.53,101.62,0.55'

        # if DATASET == "palm0.3": # 2k
        #     dset_config['roi_lon_lat_tr_lb']='101.545,0.512,101.553,0.516'
        # elif DATASET == "palm1": # 2K
        #     dset_config['roi_lon_lat_tr_lb']='101.53,0.515,101.55,0.518'
        # elif DATASET == "palm1.3": # 4k
        #     dset_config['roi_lon_lat_tr_lb']='101.52,0.515,101.555,0.518'
        if DATASET == "palm2a": # 9K
            dset_config['roi_lon_lat_tr_lb']='101.54,0.515,101.556,0.505'
        elif DATASET == "palm1a":  # 9K
            dset_config['roi_lon_lat_tr_lb'] = '101.54,0.515,101.556,0.51'
        elif DATASET == "palm2b": # 9K
            dset_config['roi_lon_lat_tr_lb']='101.54,0.505,101.556,0.495'
        elif DATASET == "palm1b":  # 9K
            dset_config['roi_lon_lat_tr_lb'] = '101.54,0.505,101.556,0.5'
        elif DATASET == "palm2c":  # 9K
            dset_config['roi_lon_lat_tr_lb']='101.556,0.515,101.572,0.505'
        elif DATASET == "palm1c":  # 9K
            dset_config['roi_lon_lat_tr_lb'] = '101.556,0.515,101.572,0.51'

        # elif DATASET == "palm2b": # 9K
        #     dset_config['roi_lon_lat_tr_lb']='101.52,0.512,101.555,0.518'
        # elif DATASET == "palm7": # 23K
        #     dset_config['roi_lon_lat_tr_lb']='101.50,0.51,101.56,0.52'
        elif DATASET == "palm10": # 44K
            dset_config['roi_lon_lat_tr_lb']='101.48,0.51,101.58,0.52'
        elif DATASET == "palm50": # 184K
            dset_config['roi_lon_lat_tr_lb']='101.45,0.5,101.62,0.525'
        elif DATASET == "palm100": # 400K
            dset_config['roi_lon_lat_tr_lb']='101.45,0.48,101.62,0.53'
    elif "olives" in DATASET:
        OBJECT = 'olives'
        dset_config[
            'LR_file'] = PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml'
        dset_config['points'] = PATH + '/barry_palm/data/labels/olives/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '-3.9,37.78,-3.79,37.9'
        dset_config['roi_lon_lat_val'] = '-3.79,37.78,-3.77,37.9'

        if DATASET == "olives100":  # 400K
            dset_config['roi_lon_lat_tr_lb'] = '-3.9,37.78,-3.79,37.9'

    elif "cars" in DATASET:
        OBJECT = 'cars'
        dset_config[
            'LR_file'] = PATH + '/barry_palm/data/2A/cars_2017/S2A_MSIL2A_20171230T183751_N0206_R027_T11SMU_20171230T202151.SAFE/MTD_MSIL2A.xml'
        dset_config['points'] = PATH + '/barry_palm/data/labels/cars/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '-117.39,34.594,-117.401,34.585'
        dset_config['roi_lon_lat_val'] = '-117.401,34.585,-117.3979,34.581'

        if DATASET == "cars100":  # 400K
            dset_config['roi_lon_lat_tr_lb'] = '-117.401,34.585,-117.39,34.594'


    else:
        print('DATASET {} Not defined'.format(DATASET))
        sys.exit(1)
    dset_config['roi_lon_lat_val_lb'] = dset_config['roi_lon_lat_val']

    dset_config['HR_file']=os.path.join(PATH,'sparse/data',OBJECT)

    if is_mounted and 'pf-pc' in socket.gethostname():
        PATH = '/scratch/andresro/leon_work'
    dset_config['save_dir']=os.path.join(PATH,'sparse/training/snapshots',OBJECT,DATASET)

    # dset_config = Namespace(**dset_config)
    return dset_config


if __name__ == '__main__':
    args = parser.parse_args()
    kml = simplekml.Kml()


    for i in ['100','50','10','2a','2b','2c','1a','1b','1c']:
        data_ = args.dataset+i
        config = get_dataset(DATASET=data_)

        for key,val in config.iteritems():

            if 'roi_lon_lat' in key:
                name_ =data_+key.replace('roi_lon_lat','_')
                pol = kml.newpolygon(name=name_)

                a = val
                roi_lon1, roi_lat1, roi_lon2, roi_lat2 = gp.split_roi_string(val)
                geo_pts = [(roi_lon1, roi_lat1), (roi_lon1, roi_lat2), (roi_lon2, roi_lat2), (roi_lon2, roi_lat1)]
                pol.outerboundaryis = geo_pts
                pol.style.polystyle.color = simplekml.Color.changealphaint(100, simplekml.Color.white)

                p1 = shapely.geometry.Polygon(geo_pts)

                geom_area = shapely.ops.transform(
                    partial(
                        pyproj.transform,
                        pyproj.Proj(init='EPSG:4326'),
                        pyproj.Proj(
                            proj='aea',
                            lat1=p1.bounds[1],
                            lat2=p1.bounds[3])),
                    p1)

                pol.description = 'Area: {:.5f} Km2'.format(geom_area.area/(1000.**2))

                print '{} Area: {:.5f} Km2'.format(name_, geom_area.area/(1000.**2))
    kml.save('roi_{}.kml'.format(args.dataset))