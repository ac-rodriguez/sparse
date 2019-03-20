import socket
import sys, os
from argparse import Namespace
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

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
        dset_config['roi_lon_lat_tr'] = '117.84,8.82,117.92,8.9'
        dset_config['roi_lon_lat_val'] = '117.81,8.82,117.84,8.88'
        dset_config['roi_lon_lat_val_lb'] = '117.81,8.82,117.84,8.88'

        if DATASET == "coco0.3": # 2K
            dset_config['roi_lon_lat_tr_lb']='117.885,8.867,117.89,8.874'
        if DATASET == "coco1":  # 4.8k
            dset_config['roi_lon_lat_tr_lb'] = '117.8821,8.8654,117.891,8.87414'
        elif DATASET == "coco2": # 8.5k
            dset_config['roi_lon_lat_tr_lb']='117.88,8.86,117.891,8.875'
        elif DATASET == "coco7": # 24k
            dset_config['roi_lon_lat_tr_lb'] = '117.87,8.86,117.9,8.88'
        elif DATASET == "coco10": # 46k
            dset_config['roi_lon_lat_tr_lb'] = '117.86,8.85,117.9,8.88'
        elif DATASET == "coco50": # 100k
            dset_config['roi_lon_lat_tr_lb'] = '117.84,8.85,117.92,8.88'
        elif DATASET == "coco100": # 267k
            dset_config['roi_lon_lat_tr_lb'] = '117.84,8.82,117.92,8.9'

    elif "palm" in DATASET:
        OBJECT='palm'
        dset_config['LR_file']=PATH+'/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml'
        dset_config['points']=PATH+'/barry_palm/data/labels/palm/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '101.45,0.48,101.62,0.53'
        dset_config['roi_lon_lat_val'] = '101.45,0.53,101.62,0.55'
        dset_config['roi_lon_lat_val_lb'] = '101.45,0.53,101.62,0.55'

        if DATASET == "palm0.3": # 2k
            dset_config['roi_lon_lat_tr_lb']='101.545,0.512,101.553,0.516'
        elif DATASET == "palm1": # 2K
            dset_config['roi_lon_lat_tr_lb']='101.53,0.515,101.55,0.518'
        elif DATASET == "palm1.3": # 4k
            dset_config['roi_lon_lat_tr_lb']='101.52,0.515,101.555,0.518'
        elif DATASET == "palm2": # 9K
            dset_config['roi_lon_lat_tr_lb']='101.52,0.512,101.555,0.518'
        elif DATASET == "palm7": # 23K
            dset_config['roi_lon_lat_tr_lb']='101.50,0.51,101.56,0.52'
        elif DATASET == "palm10": # 44K
            dset_config['roi_lon_lat_tr_lb']='101.48,0.51,101.58,0.52'
        elif DATASET == "palm50": # 184K
            dset_config['roi_lon_lat_tr_lb']='101.45,0.505,101.62,0.53'
        elif DATASET == "palm100": # 400K
            dset_config['roi_lon_lat_tr_lb']='101.45,0.48,101.62,0.53'
    elif "olives" in DATASET:
        OBJECT = 'olives'
        dset_config[
            'LR_file'] = PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml'
        dset_config['points'] = PATH + '/barry_palm/data/labels/olives/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '-3.9,37.78,-3.79,37.9'
        dset_config['roi_lon_lat_val'] = '-3.79,37.78,-3.77,37.9'
        dset_config['roi_lon_lat_val_lb'] = '-3.79,37.78,-3.77,37.9'

        if DATASET == "olives100":  # 400K
            dset_config['roi_lon_lat_tr_lb'] = '-3.9,37.78,-3.79,37.9'

    elif "cars" in DATASET:
        OBJECT = 'cars'
        dset_config[
            'LR_file'] = PATH + '/barry_palm/data/2A/cars_2017/S2A_MSIL2A_20171230T183751_N0206_R027_T11SMU_20171230T202151.SAFE/MTD_MSIL2A.xml'
        dset_config['points'] = PATH + '/barry_palm/data/labels/cars/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '-117.39,34.594,-117.401,34.585'
        dset_config['roi_lon_lat_val'] = '-117.401,34.585,-117.3979,34.581'
        dset_config['roi_lon_lat_val_lb'] = '-117.401,34.585,-117.3979,34.581'

        if DATASET == "cars100":  # 400K
            dset_config['roi_lon_lat_tr_lb'] = '-117.401,34.585,-117.39,34.594'


    else:
        print('DATASET {} Not defined'.format(DATASET))
        sys.exit(1)

    dset_config['HR_file']=os.path.join(PATH,'sparse/data',OBJECT)

    if is_mounted and 'pf-pc' in socket.gethostname():
        PATH = '/scratch/andresro/leon_work'
    dset_config['save_dir']=os.path.join(PATH,'sparse/training/snapshots',OBJECT,DATASET)

    # dset_config = Namespace(**dset_config)
    return dset_config
