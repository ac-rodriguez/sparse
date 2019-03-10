import socket
import sys, os
from argparse import Namespace
class Bunch(object):
  def __init__(self, adict):
    self.__dict__.update(adict)

def get_dataset(DATASET):

    dset_config = {}
    if 'pf-pc' in socket.gethostname():
        PATH='/home/pf/pfstaff/projects/andresro'
    else:
        PATH='/cluster/work/igp_psr/andresro'
    if DATASET == "coco":
        OBJECT='coco'
        dset_config['LR_file']=PATH+'/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml'
        dset_config['points']=PATH+'/barry_palm/data/labels/coco/points_detections.kml'
        dset_config['roi_lon_lat_tr']='117.84,8.82,117.92,8.9'
        dset_config['roi_lon_lat_tr_lb']='117.8821,8.87414,117.891,8.8654'
        dset_config['roi_lon_lat_val']='117.81,8.82,117.84,8.88'
        dset_config['roi_lon_lat_val_lb']='117.81,8.82,117.84,8.88'
    elif "palm" in DATASET:
        OBJECT='palm'
        dset_config['LR_file']=PATH+'/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml'
        dset_config['points']=PATH+'/barry_palm/data/labels/palm/kml_geoproposals'

        if DATASET == "palm":
            dset_config['roi_lon_lat_tr']='101.45,0.48,101.62,0.53'
            dset_config['roi_lon_lat_tr_lb']='101.545,0.512,101.553,0.516'
            dset_config['roi_lon_lat_val']='101.45,0.53,101.62,0.55'
            dset_config['roi_lon_lat_val_lb']='101.45,0.53,101.62,0.55'
        elif DATASET == "palm1":
            dset_config['roi_lon_lat_tr']='101.45,0.48,101.62,0.53'
            dset_config['roi_lon_lat_tr_lb']='101.55,0.51,101.58,0.52'
            dset_config['roi_lon_lat_val']='101.45,0.53,101.62,0.55'
            dset_config['roi_lon_lat_val_lb']='101.45,0.53,101.62,0.55'
        elif DATASET == "palm10":
            dset_config['roi_lon_lat_tr']='101.45,0.48,101.62,0.53'
            dset_config['roi_lon_lat_tr_lb']='101.48,0.51,101.58,0.52'
            dset_config['roi_lon_lat_val']='101.45,0.53,101.62,0.55'
            dset_config['ROI_VAL_LB']='101.45,0.53,101.62,0.55'
        elif DATASET == "palm50":
            dset_config['roi_lon_lat_tr']='101.45,0.48,101.62,0.53'
            dset_config['roi_lon_lat_tr_lb']='101.45,0.505,101.62,0.53'
            dset_config['roi_lon_lat_val']='101.45,0.53,101.62,0.55'
            dset_config['roi_lon_lat_val_lb']='101.45,0.53,101.62,0.55'
    else:
        print('DATASET {} Not defined'.format(DATASET))
        sys.exit(1)

    dset_config['HR_file']=os.path.join(PATH,'sparse/data',OBJECT)
    dset_config['save_dir']=os.path.join(PATH,'sparse/training/snapshots',DATASET)

    # dset_config = Namespace(**dset_config)
    return dset_config
