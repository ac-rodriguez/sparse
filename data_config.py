import socket
import os
import argparse
import tarfile
import glob
from osgeo import gdal
import pandas as pd
import json
import utils.gdal_processing as gp
import datetime


parser = argparse.ArgumentParser(description="Partial Supervision",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dataset", default='palm')
parser.add_argument("--save-dir", default='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/datasets/')


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
    if not isinstance(loc,list):
        locs =[loc]
    else:
        locs = loc
    filelist_out = []
    for loc in locs:
        filelist = glob.glob(PATH + f'/barry_palm/data/2A/{loc}/*_{tilename}_*.SAFE')

        filelist = glob.glob(PATH + f'/barry_palm/data/2A/{loc}/*_{tilename}_*.SAFE/MTD_MSIL2A.xml')

        lines = [line.rstrip('\n') for line in open(top_10_list)]
        lines = [x.replace('_MSIL1C_', '_MSIL2A_') + '.SAFE' for x in lines]
        filelist = [x for x in filelist if x.split('/')[-2] in lines]

        # filter that 2A is correct
        lines = [line.rstrip('\n') for line in open(PATH + f'/barry_palm/data/2A/{loc}/correct_2A.txt')]
        lines = [x + '.SAFE' for x in lines]
        filelist = [x for x in filelist if x.split('/')[-2] in lines]
        filelist_out.extend(filelist)
    return filelist_out


def get_dataset(DATASET, is_mounted=False, is_load_file=True):
    dset_config = {}
    dset_config['tr'] = []
    dset_config['val'] = []
    dset_config['test'] = []
    dset_config['is_upsample_LR'] = True  # if needed

    def add_datasets(files, gtfile, roi, datatype='tr', tilename_=''):

        if gtfile is not None and '*' in gtfile:
            gtfile = gtfile.replace('*',tilename_,1)
            if '*' in gtfile:
                gtfile = glob.glob(gtfile)
                #for f in gtfile[:]:    # Note the [:] after "files"
                #    tLog = os.path.getmtime(f)
                #    tFile = datetime.datetime.fromtimestamp(tLog)
                    #print("checking ", f, datetime.datetime.fromtimestamp(tLog))
                    #if tFile > datetime.datetime(2020,5,10):
                    #    print("time", tFile, "is bigger than 2020,5,1", "removing ", f)
                    #    gtfile.remove(f)
                    #

        if not isinstance(gtfile,list):
            gtfile = [gtfile]

        for gt_ in gtfile:
            if roi == 'geom':
                if os.path.exists(gt_):
                    roi_ = gp.get_positive_area_folder(gt_)
                else:
                    return None
            else:
                roi_ = roi

            if not isinstance(files, list):
                files = [files]
            if gt_ is not None:
                print(f'\tgt {os.path.basename(gt_)} to {len(files)} sentinel images')

            for file_ in files:
                dsREFfile = gp.get_jp2(file_, 'B03', res=10)
                dsREF = gdal.Open(dsREFfile)
                # print(gt_)
                if gp.roi_intersection(dsREF, roi_, verbose=False):
                    dset_config[datatype].append({
                        'lr': file_,
                        'hr': None,
                        'gt': gt_,
                        'roi': roi_,
                        'roi_lb': roi_,
                        'tilename': tilename_})
                else:
                    print(
                        f'dataset for {datatype} in tile {tilename_} does not intersect with roi {roi_} in {gt_}, skipping it')
    def add_datasets_intile(tilenames,rois_train=[], rois_val=[],rois_test=[], GT=None, loc=None,top_10_list=None):

        if not isinstance(tilenames, list):
            tilenames = [tilenames]
        if len(rois_train) > 0: print('Adding Train datasets')
        if len(rois_val) > 0: print('Adding Val datasets')
        if len(rois_test) > 0: print('Adding Test datasets')
        
        for tilename in tilenames:
            print('\t'+tilename)
            filelist = parse_filelist(PATH, tilename, top_10_list, loc=loc)
            if 'small' in DATASET:
                filelist = filelist[:1]
            for roi in rois_train:
                add_datasets(files=filelist, gtfile=GT, roi=roi, datatype='tr', tilename_=tilename)
            for roi in rois_val:
                add_datasets(files=filelist, gtfile=GT, roi=roi, datatype='val', tilename_=tilename)
            for roi in rois_test:
                add_datasets(files=filelist, gtfile=GT, roi=roi, datatype='test', tilename_=tilename)

    if 'pf-pc' in socket.gethostname() or 'spaceml4' in socket.gethostname() :
        # PATH = '/scratch/andresro/leon_igp'
        # if not os.path.exists(PATH):
        PATH = '/home/pf/pfstaff/projects/andresro'
        PATH_TRAIN = '/home/pf/pfstaff/projects/andresro'
    else:
        PATH = '/cluster/work/igp_psr/andresro'
        # PATH_TRAIN = '/cluster/scratch/andresro'
        PATH_TRAIN = PATH

    if is_load_file:

        filename = f'{PATH}/barry_palm/data/2A/datasets/{DATASET}.csv'
        if os.path.exists(filename):
            ds = pd.read_csv(filename)

            ds = ds.where((pd.notnull(ds)), None)
            ds['base_path'] = ds['lr'].map(lambda x: x.split('/barry_palm/', 1)[0] + '/barry_palm/')
            add_path = lambda x: PATH+'/barry_palm/'+x if isinstance(x, str) else x
            ds['lr'] = ds['lr'].map(add_path)
            ds['hr'] = ds['hr'].map(add_path)
            ds['gt'] = ds['gt'].map(add_path)


            ds_ =ds[ds.type == 'tr'][['gt', 'hr', 'lr', 'roi', 'roi_lb', 'tilename']]
            dset_config['tr'] = [row.to_dict() for index, row in ds_.iterrows()]

            ds_ = ds[ds.type == 'val'][['gt', 'hr', 'lr', 'roi', 'roi_lb', 'tilename']]
            dset_config['val'] = [row.to_dict() for index, row in ds_.iterrows()]

            ds_ = ds[ds.type == 'test'][['gt', 'hr', 'lr', 'roi', 'roi_lb', 'tilename']]
            dset_config['test'] = [row.to_dict() for index, row in ds_.iterrows()]

            with open(filename.replace('.csv','.json'), 'r') as fp:
                data = json.load(fp)
            print('loaded dataconfig from', filename)

            dset_config = {**dset_config, **data}

            save_dir = dset_config['save_dir'].split('/sparse/training/')[-1]
            dset_config['save_dir'] = PATH_TRAIN+'/sparse/training/'+ save_dir

            return dset_config
        else:
            print(f'{filename} not found...')

    if 'palmcocotiles' in DATASET:
        OBJECT = 'palmcoco'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False
        # PALM DATA

        rois = ['geom']


        # TRAIN groups1
        add_datasets_intile(['T49MCV'], rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group1',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')
        add_datasets_intile(['T49MCV'], rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/coco_annotations/*/group1',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        # VAL groups2
        add_datasets_intile(['T49MCV'], rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group2',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')
        add_datasets_intile(['T49MCV'], rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/coco_annotations/*/group2',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        if '_coco' in DATASET:
            rois_train = ['117.86,8.82,117.92,8.9']
            if '_coco1' in DATASET:  # added one negative area
                rois_train += ['117.7,8.92,117.77,8.95']
            elif '_coco1' in DATASET:  # added two neg areas
                rois_train += ['117.7,8.92,117.77,8.95', '117.57,8.85,117.61,8.83']
            add_datasets_intile('T50PNQ', rois_train=rois_train, rois_val=[], rois_test=[],
                                GT=PATH + '/barry_palm/data/labels/coco/points_detections.kml',
                                loc='phillipines_2017',
                                top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')
        if '_palm' in DATASET:
            rois_train = ['101.45,0.48,101.62,0.53']
            if '_palm1' in DATASET:  # added one negative area
                rois_train += ['101.45,0.48,101.62,0.53', '100.95,0.15,101.0,0.2']
            elif '_palm2' in DATASET:  # added two neg areas
                rois_train += ['100.95,0.15,101.0,0.2', '101.05,-0.3,101.1,-0.25']

            tilenames = ['R018_T47NQA', 'R018_T47MQV']

            add_datasets_intile(tilenames, rois_train, [], [],
                                GT=PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
                                loc='palmcountries_2017',
                                top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')
        # West Kalimantan DATA
        if '_kalim' in DATASET:
            rois_train = ['109.23,-0.85,109.63,-0.6']
            tilenames = ['R132_T49MCV']


            add_datasets_intile(tilenames, rois_train, [], [],
                                GT=PATH + '/barry_palm/data/labels/coconutSHP/Shapefile (shp)/Land Cov BPPT 2017.shp',
                                loc='palmcountries_2017',
                                top_10_list=top_10_path + '/palmcountries_2017/Indonesia_all_8410.txt')
            dset_config['attr'] = 'ID'

        if 'palmcocotiles2' in DATASET:
            tilenames_tr = ['T49NCB', 'T49NDB', 'T49NEB']

            add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                                GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                                loc='palmcountries_2017',
                                top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')


    elif 'palmcoco' in DATASET:
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
            dset_config['attr'] = 'ID'

    elif 'palmtiles' in DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        if 'palmtiles' == DATASET:
            tilenames = ['T49NCB','T49NDB','T49NEB']
        elif 'palmtiles1' == DATASET or 'palmtiles1small' == DATASET:
            tilenames = ['T49NCB','T49NDB','T49NEB','T49NFC','T49NGB']
        elif 'palmtiles2' == DATASET:
            tilenames = ['T49NFC','T49NFD','T49NGD']
        elif 'palmtiles3' == DATASET:
            tilenames = ['T49NGE','T49NHE','T49NHD','50NKL']

        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group*'
        add_datasets_intile(tilenames, rois_train=rois, rois_val=[], rois_test=[],
                            GT=GT_train,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')

        GT_val = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group*'
        add_datasets_intile(tilenames, rois_train=[], rois_val=rois, rois_test=[],
                            GT=GT_val,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')

        tilenames = ['T49NEC','T49NHD']
        add_datasets_intile(tilenames, rois_train=[], rois_val=[], rois_test=[None],
                            GT=None,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
    elif 'palmsarawak' in DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        south = ['T49NCB', 'T49NDB', 'T49NEB','T49NGB']
        north = ['T49NHD', 'T49NGE', 'T49NHE','T50NKL']
        others = ['T49NDA','T49NEC','49NED','50NKK']
        sabah = ['T50NMK', 'T50NLL', 'T50NML', 'T50NQL',
                           'T50NLM' 'T50NMM', 'T50NPM',
                            'T50NMN', '50NNN']
        if 'palmsarawak' == DATASET: #  Almost all sarawak (north & south)
            tilenames_tr = south + north
        elif 'palmsarawak1' in DATASET: # south of val set
            tilenames_tr = south
        elif 'palmsarawak2' in DATASET:  # north of val set
            tilenames_tr = north
        elif 'palmsarawak3' in DATASET: # all gt available in sarawak (north & south)
            tilenames_tr = south + north + others
        elif 'palmsarawaksabah' == DATASET: # all gt available in sarawak (north & south) + sabah (train only)
            tilenames_tr = south + north + others + sabah
        else:
            raise ValueError(DATASET+' not defined')

        # TRAIN
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
        # VAL
        tilenames_val = ['T49NFC', 'T49NFD', 'T49NGD']
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')

        # TEST
        tilenames = ['T49NEC', 'T49NHD']
        add_datasets_intile(tilenames, rois_train=[], rois_val=[], rois_test=[None],
                            GT=None,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
    elif 'palmsabah' in DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        if 'palmsabah1' in DATASET: # all gt sabah + sarawak (north)
            tilenames_tr = ['T49NHD',
                            'T49NGE', 'T49NHE','T50NKK','T50NMK',
                            'T50NKL', 'T50NLL', 'T50NML', 'T50NQL',
                            'T50NMM', 'T50NPM',
                            'T50NMN', '50NNN']
        elif 'palmsabah' in DATASET: # all gt available in sabah
            tilenames_tr = ['T50NMK',
                            'T50NKL', 'T50NLL', 'T50NML', 'T50NQL',
                            'T50NMM', 'T50NPM',
                            'T50NMN', '50NNN']
        elif 'palmsabahtest' in DATASET: # all gt available in sabah
            tilenames_tr = ['T50NMK']
        else:
            raise ValueError(DATASET+' not defined')

        # TRAIN
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group1',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group2',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')

        # VAL
        tilenames_val = ['T50NNM', 'T50NNL']
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group1',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group2',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
    elif 'palmpeninsula' in DATASET and not 'new' in DATASET:
        raise AssertionError('do not use this dataset anymore, only the new')

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        north = ['T47NRG','T47NPF','T47NRF','T47NQE']
        south = ['T47NQD','T48NTH','T48NUH','T48NUG']

        if 'palmpeninsula' == DATASET:
            tilenames_tr = north + south
        elif 'palmpeninsula1' == DATASET:
            tilenames_tr = north
        elif 'palmpeninsula2' == DATASET:
            tilenames_tr = south
        else:
            raise ValueError(DATASET+' not defined')

        # TRAIN
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')

        # VAL
        tilenames_val = ['T47NRE', 'T48NTK','T48NTJ']
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
    elif 'palmpeninsulanew' in DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        north = ['T47NRG','T47NPF','T47NRF','T47NQE','T47NRE']
        south = ['T48NTK','T48NTJ','T48NTH','T48NUG']

        if 'palmpeninsulanew' == DATASET:
            tilenames_tr = north + south
        elif 'palmpeninsulanew1' == DATASET:
            tilenames_tr = north
        elif 'palmpeninsulanew2' == DATASET:
            tilenames_tr = south
        else:
            raise ValueError(DATASET+' not defined')

        # TRAIN
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')

        # VAL
        tilenames_val = ['T47NQD', 'T48NUH']
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/palmcountries_2017/Malaysia_all_1150.txt')
    elif 'palm4748' == DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        north = ['T47NRG','T47NPF','T47NRF','T47NQE','T47NRE']
        south = ['T48NTK','T48NTJ','T48NTH','T48NUG']        
        tilenames_tr = north + south # Base tiles from Peninsula
        
        # TRAIN
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        # VAL
        tilenames_val = ['T47NQD', 'T48NUH'] # areas in peninsula
        tilenames_val += ['T47NKF', 'T47MPV', 'T48MVB'] # areas in Sumatra
        
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

    elif 'palmriau' in DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']
        riau = ['T47NPB','T47NQB','T48MTE']
        south = ['T48NTK','T48NTJ','T48NTH','T48NUG']

        if 'palmriau' == DATASET:
            tilenames_tr = riau + south
        elif 'palmriau1' == DATASET:
            tilenames_tr = riau
        else:
            raise ValueError(DATASET+' not defined')

        # TRAIN
        add_datasets_intile(tilenames_tr, rois_train=rois, rois_val=[], rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        # VAL
        tilenames_val = ['T47NQA', 'T47MQV']
        add_datasets_intile(tilenames_val, rois_train=[], rois_val=rois, rois_test=[],
                            GT=PATH + '/barry_palm/data/labels/palm_annotations/*/group*',
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

    elif 'palmborneo' in DATASET:

        OBJECT = 'palm'
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'
        dset_config['is_upsample_LR'] = False

        rois = ['geom']

        # ['Kalimantan Barat', 'Sarawak','Sabah', 'Kalimantan Utara']
        tilenames = 'T49MCV,T49MDT,T49MDU,T49MDV,T49MEV,T49NCB,T49NDA,T49NDB,T49NEB,T49NFA,T49NGA,T49NEC,T49NED,T49NFC,T49NFD,T49NGB,T49NGD,T49NGE,T49NHD,T49NHE,T50NKL,T50NLL,T50NLM,T50NMK,T50NML,T50NMM,T50NMN,T50NNK,T50NNL,T50NNM,T50NNN,T50NPM,T50NQL'
        tilenames = tilenames.split(',')

        GT_val = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group2*'
        add_datasets_intile(tilenames, rois_val=rois,
                            GT=GT_val,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        # ['Kalimantan Tengah', 'Kalimantan Selatan','Kalimantan Timur']
        tilenames = 'T49MET,T49MEU,T49MFS,T49MFT,T49MHS,T50NPG'
        tilenames = tilenames.split(',')

        GT_val = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*'
        add_datasets_intile(tilenames, rois_val=rois,
                            GT=GT_val,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        if DATASET == 'palmborneo':
            # GT before active learning
            # ['Kalimantan Barat', 'Sarawak','Sabah', 'Kalimantan Utara']
            tilenames = 'T49MCV,T49MDT,T49MDU,T49MDV,T49MEV,T49NCB,T49NDA,T49NDB,T49NEB,T49NFA,T49NGA,T49NEC,T49NED,T49NFC,T49NFD,T49NGB,T49NGD,T49NGE,T49NHD,T49NHE,T50NKL,T50NLL,T50NLM,T50NMK,T50NML,T50NMM,T50NMN,T50NNK,T50NNL,T50NNM,T50NNN,T50NPM,T50NQL'
            tilenames = tilenames.split(',')
        elif DATASET == 'palmborneo1':
            # all GT before selected active learned samples
            # ['Kalimantan', 'Sarawak','Sabah']
            tilenames = 'T49MCV,T49MDT,T49MDU,T49MEV,T49MFT,T49NDA,T49NDB,T49NEB,T49MDV,T49MET,T49MEU,T49MFS,T49MHS,T49NCB,T49NCB,T49NDB,T49NDB,T49NEC,T49NEC,T49NED,T49NED,T49NFA,T49NGA,T49NHE,T49NHE,T50MME,T50NLL,T50NLL,T50NMN,T50NMN,T50NNK,T50NNK,T50NPG,T50NQL,T50NQL,T49NFC,T49NFD,T49NGB,T49NGD,T49NGE,T49NHD,T50NKL,T50NLM,T50NMK,T50NML,T50NMM,T50NNL,T50NNM,T50NNN,T50NPM'
            tilenames = tilenames.split(',')
            GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group4*'
            add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')


        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group1*'
        add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group3*'
        add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc='palmcountries_2017',
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

    elif 'cocopalawanplus' in DATASET:
    
        OBJECT = 'coco'
        dset_config['is_upsample_LR'] = False
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'

        rois = ['geom']
        
        tilenames = 'T50PNQ,T50PPR,T50PQS,T51PUQ,T51NXH'.split(',')

        GT_val = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group2*'
        add_datasets_intile(tilenames, rois_val=rois,
                            GT=GT_val,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        if 'plus1' in DATASET:
            tilenames = tilenames + 'T51NYB,T51NXB,T51NXA'.split(',')

        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group1*'
        add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group3*'
        add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')
    elif 'cocopreactive' in DATASET:
    
        OBJECT = 'coco'
        dset_config['is_upsample_LR'] = False
        top_10_path = PATH + '/barry_palm/data/1C/dataframes_download'

        rois = ['geom']
        tilenames = 'T50PNQ,T50PNQ,T50PPR,T50PQS,T51NXA,T51NXB,T51NXG,T51NXH,T51NYB,T51NYG,T51NZH,T51NZJ,T51PUQ,T51PUR,T51PUT,T51PVR,T51PWK,T51PWQ,T51PWR,T51PYK,T51QUA'.split(',')
        # tilenames = 'T50PNQ,T50PPR,T50PQS,T51PUQ,T51NXH'.split(',')

        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group1*'
        add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        GT_train = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group3*'
        add_datasets_intile(tilenames, rois_train=rois,
                            GT=GT_train,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')

        GT_val = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group2*'
        add_datasets_intile(tilenames, rois_val=rois,
                            GT=GT_val,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')
        GT_val = f'{PATH}/barry_palm/data/labels/manual_annotations/*/{OBJECT}*group4*'
        add_datasets_intile(tilenames, rois_val=rois,
                            GT=GT_val,
                            loc=['phillipines_2017','palmcountries_2017'],
                            top_10_list=top_10_path + '/cocopalm_countries_all_11400.txt')


    #
    # elif 'cococomplete' in DATASET:
    #
    #     OBJECT = 'coco'
    #     dset_config['is_upsample_LR'] = False
    #
    #     rois_train = ['117.86,8.82,117.92,8.9', '117.7,8.92,117.77,8.95', '117.57,8.85,117.61,8.83']
    #     rois_val = ['117.84,8.82,117.86,8.88', '117.7,8.90,117.77,8.92', '117.57,8.865,117.61,8.85']
    #     rois_test = ['117.81,8.82,117.84,8.88']
    #     # rois_test = ['117.4,8.4,118.0,9.0']
    #     tilenames = ['R046_T50PNQ']
    #
    #     top_10_list = PATH + '/barry_palm/data/1C/dataframes_download/phillipines_2017/Phillipines_all_1840.txt'
    #     GT = PATH + '/barry_palm/data/labels/coco/points_detections.kml'
    #     loc = 'phillipines_2017'
    #     add_datasets_intile(tilenames, rois_train, rois_val, rois_test, GT, loc, top_10_list)
    #
    # elif "coco" in DATASET:
    #     tilename = 'R046_T50PNQ'
    #     OBJECT = 'coco'
    #     dset_config['tr'].append({
    #         'lr': PATH + '/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/coco/points_detections.kml',
    #         'roi': '117.86,8.82,117.92,8.9',
    #         'tilename': tilename})
    #
    #     dset_config['val'].append({
    #         'lr': PATH + '/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/coco/points_detections.kml',
    #         'roi': '117.84,8.82,117.86,8.88',
    #         'roi_lb': '117.84,8.82,117.86,8.88',
    #         'tilename': tilename})
    #
    #     dset_config['test'].append({
    #         'lr': PATH + '/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/coco/points_detections.kml',
    #         'roi': '117.81,8.82,117.84,8.88',
    #         'roi_lb': '117.81,8.82,117.84,8.88',
    #         'tilename': tilename})
    #
    #     if DATASET == "coco2a":  # 8.5k
    #         dset_config['tr'][0]['roi_lb'] = '117.88,8.88,117.891,8.895'
    #     elif DATASET == "coco1a":  # 8.5k
    #         dset_config['tr'][0]['roi_lb'] = '117.8821,8.8854,117.891,8.895'
    #
    #     elif DATASET == "coco2b":  # 8.5k
    #         dset_config['tr'][0]['roi_lb'] = '117.88,8.83,117.891,8.845'
    #     elif DATASET == "coco1b":  # 4.8k
    #         dset_config['tr'][0]['roi_lb'] = '117.8821,8.8354,117.891,8.845'
    #     elif DATASET == "coco2c":  # 8.5k
    #         dset_config['tr'][0]['roi_lb'] = '117.88,8.86,117.891,8.875'
    #     elif DATASET == "coco1c":  # 4.8k
    #         dset_config['tr'][0]['roi_lb'] = '117.8821,8.8654,117.891,8.875'
    #     # elif DATASET == "coco7": # 24k
    #     #     dset_config['roi_lon_lat_tr_lb'] = '117.87,8.86,117.9,8.88'
    #     elif DATASET == "coco10":  # 46k
    #         dset_config['tr'][0]['roi_lb'] = '117.86,8.85,117.9,8.88'
    #     elif DATASET == "coco50":  # 100k
    #         dset_config['tr'][0]['roi_lb'] = '117.86,8.85,117.92,8.88'
    #     elif DATASET == "coco100":  # 267k
    #         dset_config['tr'][0]['roi_lb'] = '117.86,8.82,117.92,8.9'
    #     dset_config['tr'][0]['roi'] = dset_config['tr'][0]['roi_lb']
    # elif "palmsocb" in DATASET:
    #     OBJECT = 'palm'
    #
    #     dset_config['tr'].append({
    #         'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
    #         'roi': '101.45,0.48,101.62,0.52',
    #         'roi_lb': '101.45,0.48,101.62,0.53'})
    #
    #     dset_config['val'].append({
    #         'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
    #         'roi': '101.45,0.52,101.62,0.53',
    #         'roi_lb': '101.45,0.52,101.62,0.53'})
    #
    #     dset_config['tr'].append({
    #         'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
    #         'hr': None,
    #         'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp'})
    #
    #     if DATASET == "palmsocb1":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.68'
    #     elif DATASET == "palmsocb2":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.65'
    #     elif DATASET == "palmsocb3":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.6'
    #     elif DATASET == "palmsocb4":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.54'
    #
    #     dset_config['tr'][-1]['roi_lb'] = dset_config['tr'][-1]['roi']
    #
    #     dset_config['val'].append({
    #         'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
    #         'hr': None,
    #         'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp',
    #         'roi': '-7.13,4.7,-7.1,4.54',
    #         'roi_lb': '-7.13,4.7,-7.1,4.54'})
    #     dset_config['test'].append({
    #         'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
    #         'hr': None,
    #         'gt': None,
    #         'roi': '-7.2,4.54,-7.01,4.791',
    #         'roi_lb': '-7.2,4.54,-7.01,4.791'})
    #
    #     dset_config['attr'] = 'TreeDens'
    #
    # elif "palmage" in DATASET:
    #     OBJECT = 'palm'
    #
    #     dset_config['tr'].append({
    #         'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
    #         'hr': None,
    #         'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp'})
    #
    #     if DATASET == "palmage1":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.68'
    #     elif DATASET == "palmage2":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.65'
    #     elif DATASET == "palmage3":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.6'
    #     elif DATASET == "palmage4":
    #         dset_config['tr'][-1]['roi'] = '-7.17,4.7,-7.13,4.54'
    #
    #     dset_config['tr'][-1]['roi_lb'] = dset_config['tr'][-1]['roi']
    #
    #     dset_config['val'].append({
    #         'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
    #         'hr': None,
    #         'gt': PATH + '/barry_palm/data/labels/socb/palm_density.shp',
    #         'roi': '-7.13,4.7,-7.1,4.54',
    #         'roi_lb': '-7.13,4.7,-7.1,4.54'})
    #     dset_config['test'].append({
    #         'lr': PATH + '/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE',
    #         'hr': None,
    #         'gt': None,
    #         'roi': '-7.2,4.54,-7.01,4.791',
    #         'roi_lb': '-7.2,4.54,-7.01,4.791'})
    # elif "palm" in DATASET:
    #     OBJECT = 'palm'
    #     dset_config['tr'].append({
    #         'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
    #         'roi': '101.45,0.48,101.62,0.52'})
    #
    #     dset_config['val'].append({
    #         'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
    #         'roi': '101.45,0.52,101.62,0.53',
    #         'roi_lb': '101.45,0.52,101.62,0.53'})
    #     dset_config['test'].append({
    #         'lr': PATH + '/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/palm/kml_geoproposals',
    #         'roi': '101.45,0.53,101.62,0.55',
    #         'roi_lb': '101.45,0.53,101.62,0.55'})
    #
    #     if DATASET == "palm2a":  # 9K
    #         dset_config['tr'][0]['roi_lb'] = '101.54,0.515,101.556,0.505'
    #     elif DATASET == "palm1a":  # 9K
    #         dset_config['tr'][0]['roi_lb'] = '101.54,0.515,101.556,0.51'
    #     elif DATASET == "palm2b":  # 9K
    #         dset_config['tr'][0]['roi_lb'] = '101.54,0.505,101.556,0.495'
    #     elif DATASET == "palm1b":  # 9K
    #         dset_config['tr'][0]['roi_lb'] = '101.54,0.505,101.556,0.5'
    #     elif DATASET == "palm2c":  # 9K
    #         dset_config['tr'][0]['roi_lb'] = '101.556,0.515,101.572,0.505'
    #     elif DATASET == "palm1c":  # 9K
    #         dset_config['tr'][0]['roi_lb'] = '101.556,0.515,101.572,0.51'
    #
    #     elif DATASET == "palm10":  # 44K
    #         dset_config['tr'][0]['roi_lb'] = '101.48,0.51,101.58,0.52'
    #     elif DATASET == "palm50":  # 184K
    #         dset_config['tr'][0]['roi_lb'] = '101.45,0.5,101.62,0.525'
    #     elif DATASET == "palm100":  # 400K
    #         dset_config['tr'][0]['roi_lb'] = '101.45,0.48,101.62,0.53'
    # elif "olives" in DATASET:
    #     OBJECT = 'olives'
    #
    #     dset_config['tr'].append({
    #         'lr': PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/olives/kml_geoproposals',
    #         'roi': '-3.9,37.78,-3.79,37.9'})
    #     dset_config['val'].append({
    #         'lr': PATH + '/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml',
    #         'hr': os.path.join(PATH, 'sparse/data', OBJECT),
    #         'gt': PATH + '/barry_palm/data/labels/olives/kml_geoproposals',
    #         'roi': '-3.79,37.78,-3.77,37.9'})
    #
    #     if DATASET == "olives100":  # 400K
    #         dset_config['tr'][0]['roi_lb'] = '-3.9,37.78,-3.79,37.9'
    #
    # elif "cars" in DATASET:
    #     OBJECT = 'cars'
    #     dset_config[
    #         'LR_file'] = PATH + '/barry_palm/data/2A/cars_2017/S2A_MSIL2A_20171230T183751_N0206_R027_T11SMU_20171230T202151.SAFE/MTD_MSIL2A.xml'
    #     dset_config['points'] = PATH + '/barry_palm/data/labels/cars/kml_geoproposals'
    #     dset_config['roi_lon_lat_tr'] = '-117.39,34.594,-117.401,34.585'
    #     dset_config['roi_lon_lat_val'] = '-117.401,34.585,-117.3979,34.581'
    #
    #     if DATASET == "cars100":  # 400K
    #         dset_config['roi_lon_lat_tr_lb'] = '-117.401,34.585,-117.39,34.594'

    else:
        raise ValueError('DATASET {} Not defined'.format(DATASET))

    if is_mounted and 'pf-pc' in socket.gethostname():
        PATH_TRAIN = '/scratch/andresro/leon_work'
    dset_config['save_dir'] = os.path.join(PATH_TRAIN, 'sparse/training', OBJECT, DATASET)

    return dset_config


if __name__ == '__main__':

    args = parser.parse_args()

    config = get_dataset(args.dataset, is_load_file=False)


    ds_train = pd.DataFrame(config['tr'])
    ds_train['type'] = 'tr'

    ds_val = pd.DataFrame(config['val'])
    ds_val['type'] = 'val'

    ds_test = pd.DataFrame(config['test'])
    ds_test['type'] = 'test'

    ds = pd.concat((ds_train,ds_val,ds_test),axis=0)

    ds['base_path'] = ds['lr'].map(lambda x: x.split('/barry_palm/',1)[0]+'/barry_palm/')
    f_parse = lambda x: x.split('/barry_palm/',1)[-1] if isinstance(x,str) else x
    ds['lr'] = ds['lr'].map(f_parse)
    ds['hr'] = ds['hr'].map(f_parse)
    ds['gt'] = ds['gt'].map(f_parse)

    filename = f"{args.save_dir}/{args.dataset}.csv"
    ds.to_csv(filename)

    config.pop('tr')
    config.pop('val')
    config.pop('test')
    filename = f"{args.save_dir}/{args.dataset}.json"

    with open(filename, 'w') as fp:
        json.dump(config, fp)
    print('saved', filename)
