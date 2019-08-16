import socket
import sys, os
import argparse
import simplekml
import shapely
from functools import partial
import tarfile
import pyproj
import glob
from osgeo import gdal
import gdal_processing as gp

parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", default='palm')


def untar(file_pattern, save_dir=None):
    if save_dir is None:
        save_dir = os.path.dirname(file_pattern)
    if file_pattern.endswith('.SAFE'):
        file_pattern = file_pattern.replace('.SAFE', '.tar')
    elif not file_pattern.endswith('.tar'):
        raise ValueError('wrong file name', file_pattern)
    # TODO extract only the ones missing
    filelist = glob.glob(file_pattern)
    print('extracting SAFE from tar files')
    for file in filelist:
        tar = tarfile.open(file)
        tar.extractall(path=save_dir)
        tar.close()


def parse_filelist(PATH, tilename, top_10_list, loc):
    filelist = glob.glob(PATH + f'/barry_palm/data/2A/{loc}/*_{tilename}_*.SAFE')

    if len(filelist) < 10:
        # TODO extract only the ones missing
        file_pattern = PATH + f'/barry_palm/data/2A/{loc}/*{tilename}*.tar'
        untar(file_pattern, save_dir=PATH + f'/barry_palm/data/2A/{loc}/')

    filelist = glob.glob(PATH + f'/barry_palm/data/2A/{loc}/*_{tilename}_*.SAFE/MTD_MSIL2A.xml')

    lines = [line.rstrip('\n') for line in open(top_10_list)]
    lines = [x.replace('_MSIL1C_', '_MSIL2A_') + '.SAFE' for x in lines]
    filelist = [x for x in filelist if x.split('/')[-2] in lines]

    # filter that 2A is correct
    lines = [line.rstrip('\n') for line in open(PATH + f'/barry_palm/data/2A/{loc}/correct_2A.txt')]
    lines = [x + '.SAFE' for x in lines]
    filelist = [x for x in filelist if x.split('/')[-2] in lines]
    return filelist


def get_dataset(DATASET, is_mounted=False):
    dset_config = {}
    dset_config['tr'] = []
    dset_config['val'] = []
    dset_config['test'] = []
    dset_config['is_upsample_LR'] = True  # if needed

    def add_datasets(files, gtfile, roi_, datatype='tr', tilename_=''):
        if not isinstance(files, list):
            files = [files]
        for file_ in files:
            dsREFfile = gp.get_jp2(file_, 'B03', res=10)
            dsREF = gdal.Open(dsREFfile)

            if gp.roi_intersection(dsREF, roi_):
                dset_config[datatype].append({
                    'lr': file_,
                    'hr': None,
                    'gt': gtfile,
                    'roi': roi_,
                    'roi_lb': roi_,
                    'tilename': tilename_})
            else:
                print(f'dataset for {datatype} in tile {tilename_} does not intersect with roi {roi_}, skipping it')
    def add_datasets_intile(tilenames,rois_train, rois_val,rois_test, GT, loc,top_10_list):

        if DATASET.endswith('A'):
            rois_train = rois_train[:1]
            rois_val = rois_val[:1]
            rois_test = rois_test[:1]
            tilenames = tilenames[:1]

        for tilename in tilenames:
            filelist = parse_filelist(PATH, tilename, top_10_list, loc=loc)
            if 'small' in DATASET:
                filelist = filelist[:1]
            print('Adding Train datasets')
            for roi in rois_train:
                add_datasets(files=filelist, gtfile=GT, roi_=roi, datatype='tr', tilename_=tilename)
            print('Adding Val datasets')
            for roi in rois_val:
                add_datasets(files=filelist, gtfile=GT, roi_=roi, datatype='val', tilename_=tilename)
            print('Adding Test datasets')
            for roi in rois_test:
                add_datasets(files=filelist, gtfile=GT, roi_=roi, datatype='test', tilename_=tilename)

    if 'pf-pc' in socket.gethostname():
        # PATH='/home/pf/pfstaff/projects/andresro'
        PATH = '/scratch/andresro/leon_igp'
        PATH_TRAIN = '/home/pf/pfstaff/projects/andresro'
    else:
        PATH = '/cluster/work/igp_psr/andresro'
        PATH_TRAIN = '/cluster/scratch/andresro'
    if 'palmcoco' in DATASET:
        OBJECT = 'palmcoco'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False
        # PALM DATA

        rois_train = ['101.45,0.48,101.62,0.53','100.95,0.15,101.0,0.2', '101.05,-0.3,101.1,-0.25']  # was 0.52
        rois_val = ['101.45,0.52,101.62,0.53']
        rois_test = ['101.45,0.53,101.62,0.55']
        tilenames = ['R018_T47NQA','R018_T47MQV']


        add_datasets_intile(tilenames, rois_train, rois_val, rois_test,
                            GT=PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Indonesia_all_8410.txt')

        # COCO DATA

        # OBJECT = 'coco'

        rois_train = ['117.86,8.82,117.92,8.9', '117.7,8.92,117.77,8.95', '117.57,8.85,117.61,8.83']
        rois_val = ['117.84,8.82,117.86,8.88', '117.7,8.90,117.77,8.92', '117.57,8.865,117.61,8.85']
        rois_test = ['117.81,8.82,117.84,8.88']
        tilenames = ['T50PNQ']

        add_datasets_intile(tilenames, rois_train, rois_val, rois_test,
                            GT=PATH + '/barry_palm/data/labels/coco/points_detections.kml',
                            loc='phillipines_2017',
                            top_10_list=top_10_path + '/phillipines_2017/Phillipines_all_1840.txt')

        # West Kalimantan DATA
        if 'kalim' in DATASET:
            rois_train = ['109.23,-0.85,109.63,-0.6']
            rois_val = ['109.24,-0.85,109.63,-0.93']
            rois_test = ['109.2,-1.0,109.8,-0.6']
            tilenames = ['R132_T49MCV']


            add_datasets_intile(tilenames, rois_train, rois_val, rois_test,
                                GT=PATH + '/barry_palm/data/labels/coconutSHP/Shapefile (shp)/Land Cov BPPT 2017.shp',
                                loc='palmcountries_2017',
                                top_10_list=top_10_path + '/palmcountries_2017/Indonesia_all_8410.txt')


    elif 'cococomplete' in DATASET:

        OBJECT = 'coco'
        dset_config['is_upsample_LR'] = False

        rois_train = ['117.86,8.82,117.92,8.9', '117.7,8.92,117.77,8.95', '117.57,8.85,117.61,8.83']
        rois_val = ['117.84,8.82,117.86,8.88', '117.7,8.90,117.77,8.92', '117.57,8.865,117.61,8.85']
        rois_test = ['117.81,8.82,117.84,8.88']
        # rois_test = ['117.4,8.4,118.0,9.0']
        tilenames = ['T50PNQ']

        top_10_list = PATH + '/barry_palm/data/1C/dataframes_download/phillipines_2017/Phillipines_all_1840.txt'
        GT = PATH + '/barry_palm/data/labels/coco/points_detections.kml'
        loc = 'phillipines_2017'
        add_datasets_intile(tilenames, rois_train, rois_val, rois_test, GT, loc, top_10_list)

    elif "coco" in DATASET:
        tilename = 'R046_T50PNQ'
        OBJECT = 'coco'
        dset_config['tr'].append({
            'lr': PATH + '/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/coco/points_detections.kml',
            'roi': '117.86,8.82,117.92,8.9',
            'tilename': tilename})

        dset_config['val'].append({
            'lr': PATH + '/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/coco/points_detections.kml',
            'roi': '117.84,8.82,117.86,8.88',
            'roi_lb': '117.84,8.82,117.86,8.88',
            'tilename': tilename})

        dset_config['test'].append({
            'lr': PATH + '/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/coco/points_detections.kml',
            'roi': '117.81,8.82,117.84,8.88',
            'roi_lb': '117.81,8.82,117.84,8.88',
            'tilename': tilename})

        if DATASET == "coco2a":  # 8.5k
            dset_config['tr'][0]['roi_lb'] = '117.88,8.88,117.891,8.895'
        elif DATASET == "coco1a":  # 8.5k
            dset_config['tr'][0]['roi_lb'] = '117.8821,8.8854,117.891,8.895'

        elif DATASET == "coco2b":  # 8.5k
            dset_config['tr'][0]['roi_lb'] = '117.88,8.83,117.891,8.845'
        elif DATASET == "coco1b":  # 4.8k
            dset_config['tr'][0]['roi_lb'] = '117.8821,8.8354,117.891,8.845'
        elif DATASET == "coco2c":  # 8.5k
            dset_config['tr'][0]['roi_lb'] = '117.88,8.86,117.891,8.875'
        elif DATASET == "coco1c":  # 4.8k
            dset_config['tr'][0]['roi_lb'] = '117.8821,8.8654,117.891,8.875'
        # elif DATASET == "coco7": # 24k
        #     dset_config['roi_lon_lat_tr_lb'] = '117.87,8.86,117.9,8.88'
        elif DATASET == "coco10":  # 46k
            dset_config['tr'][0]['roi_lb'] = '117.86,8.85,117.9,8.88'
        elif DATASET == "coco50":  # 100k
            dset_config['tr'][0]['roi_lb'] = '117.86,8.85,117.92,8.88'
        elif DATASET == "coco100":  # 267k
            dset_config['tr'][0]['roi_lb'] = '117.86,8.82,117.92,8.9'
        dset_config['tr'][0]['roi'] = dset_config['tr'][0]['roi_lb']
    elif "palmsocb" in DATASET:
        OBJECT = 'palm'

        dset_config['tr'].append({
            'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
            'roi': '101.45,0.48,101.62,0.52',
            'roi_lb': '101.45,0.48,101.62,0.53'})

        dset_config['val'].append({
            'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
            'roi': '101.45,0.52,101.62,0.53',
            'roi_lb': '101.45,0.52,101.62,0.53'})

        dset_config['tr'].append({
            'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
            'hr': None,
            'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp'})

        if DATASET == "palmsocb1":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.68'
        elif DATASET == "palmsocb2":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.65'
        elif DATASET == "palmsocb3":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.6'
        elif DATASET == "palmsocb4":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.54'

        dset_config['tr'][-1]['roi_lb'] = dset_config['tr'][-1]['roi']

        dset_config['val'].append({
            'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
            'hr': None,
            'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp',
            'roi': '-7.13,4.7,-7.1,4.54',
            'roi_lb': '-7.13,4.7,-7.1,4.54'})
        dset_config['test'].append({
            'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
            'hr': None,
            'gt': None,
            'roi': '-7.2,4.54,-7.01,4.791',
            'roi_lb': '-7.2,4.54,-7.01,4.791'})

        dset_config['attr'] = 'TreeDens'

    elif "palmage" in DATASET:
        OBJECT = 'palm'

        dset_config['tr'].append({
            'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
            'hr': None,
            'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp'})

        if DATASET == "palmage1":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.68'
        elif DATASET == "palmage2":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.65'
        elif DATASET == "palmage3":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.6'
        elif DATASET == "palmage4":
            dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.54'

        dset_config['tr'][-1]['roi_lb'] = dset_config['tr'][-1]['roi']

        dset_config['val'].append({
            'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
            'hr': None,
            'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp',
            'roi': '-7.13,4.7,-7.1,4.54',
            'roi_lb': '-7.13,4.7,-7.1,4.54'})
        dset_config['test'].append({
            'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
            'hr': None,
            'gt': None,
            'roi': '-7.2,4.54,-7.01,4.791',
            'roi_lb': '-7.2,4.54,-7.01,4.791'})

        # dset_config['tr'].append({
        #     'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
        #     'hr': None,
        #     'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp',
        #     'roi': '-7.2,4.5,-7.1,4.58',
        #     'roi_lb': '-7.2,4.5,-7.1,4.58'})
        #
        # dset_config['val'].append({
        #     'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
        #     'hr': None,
        #     'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp',
        #     'roi': '-7.2,4.58,-7.1,4.65',
        #     'roi_lb': '-7.2,4.58,-7.1,4.65'})
        # dset_config['test'].append({
        #     'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
        #     'hr': None,
        #     'gt': None,
        #     'roi': '-7.2,4.5,-7.1,4.58',
        #     'roi_lb': '-7.2,4.5,-7.1,4.58'})
        dset_config['attr'] = 'TreeAge'


    elif "palm" in DATASET:
        OBJECT = 'palm'
        dset_config['tr'].append({
            'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
            'roi': '101.45,0.48,101.62,0.52'})

        dset_config['val'].append({
            'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
            'roi': '101.45,0.52,101.62,0.53',
            'roi_lb': '101.45,0.52,101.62,0.53'})
        dset_config['test'].append({
            'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
            'roi': '101.45,0.53,101.62,0.55',
            'roi_lb': '101.45,0.53,101.62,0.55'})
        # dset_config['LR_file']=PATH+'/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml'
        # dset_config['points']=PATH+'/barry_palm/data/labels/palm/kml_geoproposals'
        # dset_config['roi_lon_lat_tr'] = '101.45,0.48,101.62,0.52'
        # dset_config['roi_lon_lat_val'] = '101.45,0.52,101.62,0.53'
        # dset_config['roi_lon_lat_test'] = '101.45,0.53,101.62,0.55'

        # if DATASET == "palm0.3": # 2k
        #     dset_config['roi_lon_lat_tr_lb']='101.545,0.512,101.553,0.516'
        # elif DATASET == "palm1": # 2K
        #     dset_config['roi_lon_lat_tr_lb']='101.53,0.515,101.55,0.518'
        # elif DATASET == "palm1.3": # 4k
        #     dset_config['roi_lon_lat_tr_lb']='101.52,0.515,101.555,0.518'
        if DATASET == "palm2a":  # 9K
            dset_config['tr'][0]['roi_lb'] = '101.54,0.515,101.556,0.505'
        elif DATASET == "palm1a":  # 9K
            dset_config['tr'][0]['roi_lb'] = '101.54,0.515,101.556,0.51'
        elif DATASET == "palm2b":  # 9K
            dset_config['tr'][0]['roi_lb'] = '101.54,0.505,101.556,0.495'
        elif DATASET == "palm1b":  # 9K
            dset_config['tr'][0]['roi_lb'] = '101.54,0.505,101.556,0.5'
        elif DATASET == "palm2c":  # 9K
            dset_config['tr'][0]['roi_lb'] = '101.556,0.515,101.572,0.505'
        elif DATASET == "palm1c":  # 9K
            dset_config['tr'][0]['roi_lb'] = '101.556,0.515,101.572,0.51'

        # elif DATASET == "palm2b": # 9K
        #     dset_config['roi_lon_lat_tr_lb']='101.52,0.512,101.555,0.518'
        # elif DATASET == "palm7": # 23K
        #     dset_config['roi_lon_lat_tr_lb']='101.50,0.51,101.56,0.52'
        elif DATASET == "palm10":  # 44K
            dset_config['tr'][0]['roi_lb'] = '101.48,0.51,101.58,0.52'
        elif DATASET == "palm50":  # 184K
            dset_config['tr'][0]['roi_lb'] = '101.45,0.5,101.62,0.525'
        elif DATASET == "palm100":  # 400K
            dset_config['tr'][0]['roi_lb'] = '101.45,0.48,101.62,0.53'
    elif "olives" in DATASET:
        OBJECT = 'olives'

        dset_config['tr'].append({
            'lr': PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/olives/kml_geoproposals',
            'roi': '-3.9,37.78,-3.79,37.9'})
        dset_config['val'].append({
            'lr': PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml',
            'hr': os.path.join(PATH, 'sparse/data', OBJECT),
            'gt': PATH + '/barry_palm/data/labels/olives/kml_geoproposals',
            'roi': '-3.79,37.78,-3.77,37.9'})
        # dset_config[
        #     'LR_file'] = PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml'
        # dset_config['points'] = PATH + '/barry_palm/data/labels/olives/kml_geoproposals'
        # dset_config['roi_lon_lat_tr'] = '-3.9,37.78,-3.79,37.9'
        # dset_config['roi_lon_lat_val'] = '-3.79,37.78,-3.77,37.9'
        if DATASET == "olives100":  # 400K
            dset_config['tr'][0]['roi_lb'] = '-3.9,37.78,-3.79,37.9'

    elif "cars" in DATASET:
        OBJECT = 'cars'
        dset_config[
            'LR_file'] = PATH + '/barry_palm/data/2A/cars_2017/S2A_MSIL2A_20171230T183751_N0206_R027_T11SMU_20171230T202151.SAFE/MTD_MSIL2A.xml'
        dset_config['points'] = PATH + '/barry_palm/data/labels/cars/kml_geoproposals'
        dset_config['roi_lon_lat_tr'] = '-117.39,34.594,-117.401,34.585'
        dset_config['roi_lon_lat_val'] = '-117.401,34.585,-117.3979,34.581'

        if DATASET == "cars100":  # 400K
            dset_config['roi_lon_lat_tr_lb'] = '-117.401,34.585,-117.39,34.594'

    elif "vaihingen" in DATASET:
        # TODO add test dataset
        OBJECT = 'vaihingen'
        dset_config[
            'LR_file'] = None
        dset_config['roi_lon_lat_tr'] = None
        dset_config['roi_lon_lat_val'] = None
        dset_config['roi_lon_lat_tr_lb'] = None
        if DATASET == 'vaihingensmall':
            path_ = PATH + '/sparse/data/vaihingen/small/'

            dset_config['tr'].append({
                'hr': path_ + 'top_9cm_area1.tif',
                'dsm': path_ + 'dsm_9cm_area1.tif',
                'sem': path_ + 'sem_9cm_area1.tif'})
            dset_config['val'].append({
                'hr': path_ + 'top_9cm_area4.tif',
                'dsm': path_ + 'dsm_9cm_area4.tif',
                'sem': path_ + 'sem_9cm_area4.tif'})

        elif DATASET == 'vaihingenComplete':
            path_ = PATH + '/sparse/data/vaihingen/'

            dset_config['sem_train'] = path_ + 'sem_train_9cm.tif'
            dset_config['dsm_train'] = path_ + 'dsm_train_9cm.tif'
            dset_config['top_train'] = path_ + 'top_train_9cm.tif'

            dset_config['sem_val'] = path_ + 'sem_val_9cm.tif'
            dset_config['dsm_val'] = path_ + 'dsm_val_9cm.tif'
            dset_config['top_val'] = path_ + 'top_val_9cm.tif'

            dset_config['sem_test'] = path_ + 'sem_9cm.tif'
            dset_config['dsm_test'] = path_ + 'dsm_9cm.tif'
            dset_config['top_test'] = path_ + 'top_9cm.tif'
        else:
            path_ = PATH + '/sparse/data/ftp.ipi.uni-hannover.de/ISPRS_BENCHMARK_DATASETS/Vaihingen/semantic_segmentation_labeling_Vaihingen/'

            for file in sorted(glob.glob(path_ + "dsm_train/*.tif")):
                file_number = file.split('area')[-1]
                dset_config['tr'].append({
                    'hr': glob.glob(f"{path_}top_train/*{file_number}")[0],
                    'dsm': glob.glob(f"{path_}dsm_train/*{file_number}")[0],
                    'sem': glob.glob(f"{path_}gt_train/*{file_number}")[0]})

            for file in sorted(glob.glob(path_ + "dsm_val/*.tif")):
                file_number = file.split('area')[-1]
                dset_config['val'].append({
                    'hr': glob.glob(f"{path_}top_val/*{file_number}")[0],
                    'dsm': glob.glob(f"{path_}dsm_val/*{file_number}")[0],
                    'sem': glob.glob(f"{path_}gt_val/*{file_number}")[0]})
            # dset_config['val'] = [dset_config['val'][0]]
            # dset_config['val'].append({
            #     'sem': path_ + 'sem_val_9cm.tif',
            #     'dsm': path_ + 'dsm_val_9cm.tif',
            #     'hr': path_ + 'top_val_9cm.tif'})
            path_ = PATH + '/sparse/data/vaihingen/'
            dset_config['test'].append({
                'sem': path_ + 'sem_9cm.tif',
                'dsm': path_ + 'dsm_9cm.tif',
                'hr': path_ + 'top_9cm.tif'})
            if not DATASET.endswith('gen'):
                data_size = int(DATASET.split('en')[-1])
                dset_config['tr'] = dset_config['tr'][:data_size]
                # dset_config['val'] = dset_config['val'][:data_size]

        dset_config['is_noS2'] = True
    else:
        raise ValueError('DATASET {} Not defined'.format(DATASET))
    # dset_config['roi_lon_lat_val_lb'] = dset_config['roi_lon_lat_val']

    # dset_config['HR_file']=os.path.join(PATH,'sparse/data',OBJECT)

    if is_mounted and 'pf-pc' in socket.gethostname():
        PATH_TRAIN = '/scratch/andresro/leon_work'
    dset_config['save_dir'] = os.path.join(PATH_TRAIN, 'sparse/training/snapshots', OBJECT, DATASET)

    # dset_config = Namespace(**dset_config)
    return dset_config


if __name__ == '__main__':
    args = parser.parse_args()
    kml = simplekml.Kml()

    for i in ['100', '50', '10', '2a', '2b', '2c', '1a', '1b', '1c']:
        data_ = args.dataset + i
        config = get_dataset(DATASET=data_)

        for key, val in config.iteritems():

            if 'roi_lon_lat' in key:
                name_ = data_ + key.replace('roi_lon_lat', '_')
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

                pol.description = 'Area: {:.5f} Km2'.format(geom_area.area / (1000. ** 2))

                print('{} Area: {:.5f} Km2'.format(name_, geom_area.area / (1000. ** 2)))
    kml.save('roi_{}.kml'.format(args.dataset))
