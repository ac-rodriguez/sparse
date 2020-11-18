import argparse
import sys
from sentinelsat import SentinelAPI, InvalidChecksumError, SentinelAPILTAError, SentinelAPIError
import glob
import os
import numpy as np
import pandas as pd
import json
import shutil
import tarfile
# import h5py
# from osgeo import gdal

parser = argparse.ArgumentParser(description="Download Sentinel-2 tiles",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--username", default=None)
parser.add_argument("--userid", default=0, type=int)
parser.add_argument("--password", default='ecovision')
parser.add_argument("--df-path", default=None)
parser.add_argument("--save-dir", default=None)
parser.add_argument("--is-offline-only", default=False, action="store_true",
                    help="use wget to download")
parser.add_argument("--skip-failed", default=False, action="store_true",
                    help="use wget to download")
parser.add_argument("--is-wget", default=False, action="store_true",
                    help="use wget to download")
parser.add_argument("--is-checkonly", default=False, action="store_true",
                    help="use wget to download")
parser.add_argument("--is-checksum", default=False, action="store_true",
                    help="check MD5 sum from downloaded")
parser.add_argument("--is-split", default='1-1', help='n-N split in N parts and process part n')
parser.add_argument("--is-check2A", default=False, action="store_true",
                    help="checksum .jp2 of 2A folders")
parser.add_argument("--is-cleanup", default=False, action="store_true",
                    help="remove 1C safe files after 2A was checked")
parser.add_argument("--is-delete", default=False, action="store_true",
                    help="delete corresponding check operation")
parser.add_argument("--is-tar2A", default=False, action="store_true",
                    help="tar 2A datasets per orbit_tile numbers")
parser.add_argument("--is-h52A", default=False, action="store_true",
                    help="create h5 2A datasets per orbit_tile numbers")
args = parser.parse_args()
print(' '.join(sys.argv))
df = pd.read_pickle(args.df_path)
N = df.shape[0]

part, splits = [int(x) for x in args.is_split.split('-')]

N_splits = N // splits

index_start, index_end = (part - 1) * N_splits, part * N_splits
if part == splits:
    index_end = N
df = df[index_start:index_end]
print(N, 'total files')
print(f' processing part {part}/{splits}')
print(f' subset from {index_start}:{index_end}')

print(args.save_dir)
filelist = glob.glob(args.save_dir + '/*.zip')

existing_ds = [os.path.split(x)[-1].replace('.zip', '') for x in filelist]

## do not redownload if in correct_zip
file_existing_correct = args.save_dir + '/correct_zip.txt'
if os.path.isfile(file_existing_correct):
    lines = [line.rstrip('\n') for line in open(file_existing_correct)]
    print(len(lines),'in correct_zip')
    print(len(existing_ds),'existing zip')
    existing_ds = set(existing_ds+lines)
    print(len(existing_ds),'existing ds')
    # sys.exit(0)

is_pending = [x not in existing_ds for x in df.title]
print('total', df.shape[0])
print('existing in folder', len(existing_ds))
print('existing in area', len(is_pending) - np.sum(is_pending))
print('pending for download', np.sum(is_pending))


username_list = ['andrescamilor','andresro','sdaronco','nassar','lasinger','ceneka' ,'manutom.k']

from datetime import datetime
import time

last_usedtime = [datetime(2020,1,1)]*len(username_list)
current_user = args.userid
last_usedtime[current_user] = datetime.now()

args.username = username_list[args.userid]

def get_password(username):
    if username == 'manutom.k':
        return 'manu1234'
    else:
        return 'ecovision'


def change_user(wait_time = 30.0):
    now = datetime.now()
    current_time_dif = [(now-x).total_seconds() / 60.0 for x in last_usedtime]
    if np.max(current_time_dif) < wait_time:
        min_to_wait = wait_time+0.1 - np.max(current_time_dif)
        print(f'waiting for {min_to_wait} min')
        time.sleep(60*min_to_wait)

    new_user = np.argmax(current_time_dif)
    print('changed to user ',username_list[new_user])
    last_usedtime[new_user] = datetime.now()

    return new_user


if args.username == 'manutom.k':
    args.passwort = 'manu1234'

api = SentinelAPI(args.username, get_password(args.username), 'https://scihub.copernicus.eu/dhus')

if args.is_checksum:
    file_existing_correct = args.save_dir + '/correct_zip.txt'
    df_existing = df[~np.array(is_pending)]

    # Check if file was already checkedsum

    if os.path.isfile(file_existing_correct):
        lines = [line.rstrip('\n') for line in open(file_existing_correct)]

        is_not_checked = [x not in lines for x in df_existing.title]
        print(np.sum(~np.array(is_not_checked)), 'files already checksum checked ')
        df_existing = df_existing[is_not_checked]

    index_ = list(df_existing.index)
    if len(index_) > 0:
        print(f'checksum for {len(index_)} files pending')
        dict_missing = api.check_files(ids=index_, directory=args.save_dir + "/", delete=args.is_delete)
        correct_titles = [x for x in df_existing.title if x not in dict_missing.keys()]

        with open(file_existing_correct, 'a+') as f:
            for item in correct_titles:
                f.write("%s\n" % item)
        with open(args.save_dir + '/incorrect_zip.json', 'a+') as fp:
            json.dump(dict_missing, fp, indent=4, sort_keys=True, default=str)

        # Correct pending_ds with incorrect files
        incorrect_keys = set(dict_missing.keys())
        is_incorrect = [x in incorrect_keys for x in df.title]
        is_pending = np.logical_or(is_pending, is_incorrect)

df1 = df[is_pending]
if not args.is_checkonly:

    index_failed = []
    for counter, (index, row) in enumerate(df1.iterrows()):
        while True:
            print(counter)
            try:
                # if not args.is_offline_only:
                # if api.get_product_odata(index)['Online'] and not args.is_offline_only:
                if api.get_product_odata(index)['Online']:

                    print('downloading {}/{}    {}'.format(counter, df1.shape[0], row.title))
                    if args.is_wget:
                        command = f'wget --content-disposition --continue ' \
                            f'--user={args.username} ' \
                            f'--password={args.password}' \
                            f' "https://scihub.copernicus.eu/dhus/odata/v1/Products(\'{index}\')/\$value" ' \
                            f'-P {args.save_dir}'
                        os.system(command)
                    else:
                        response = api.download(index, directory_path=args.save_dir)
                        if not response['Online']:
                            print(row.title, 'will re-try at the end')
                            index_failed.append(index)

                elif args.is_offline_only:
                    print('not online')
                    response = api.download(index, directory_path=args.save_dir)
                    if not response['Online']:
                        print(row.title, 'will re-try at the end')
                        index_failed.append(index)

            except SentinelAPILTAError:
                current_user = change_user(wait_time=30.0)
                api = SentinelAPI(username_list[current_user], get_password(username_list[current_user]), 'https://scihub.copernicus.eu/dhus')
                if not args.skip_failed:
                    continue
            except SentinelAPIError:
                # print('too many requests, waiting 10 min')
                # time.sleep(10*60)
                current_user = change_user(wait_time=10.0)
                api = SentinelAPI(username_list[current_user], get_password(username_list[current_user]), 'https://scihub.copernicus.eu/dhus')
                if not args.skip_failed:
                    continue
            # except (SentinelAPILTAError,SentinelAPIError) as e:
            #     if args.is_offline_only:
            #         # index_failed.append(index)
            #         current_user = change_user(wait_time=30.0)
            #         # print('limit reached, changing user')                
            #         api = SentinelAPI(username_list[current_user], get_password(username_list[current_user]), 'https://scihub.copernicus.eu/dhus')
            #     else:
            #         current_user = change_user(wait_time=10.0)
            #         # print('too many requests, waiting 10 min')
            #         # time.sleep(10*60)
            except InvalidChecksumError:
                print(row.title, 'failed checksum, will re-try at the end')
                index_failed.append(index)
            break
        

    if len(index_failed) > 0:
        print('Retrying failed {} datasets..'.format(len(index_failed)))
        for i in index_failed:
            api.download(i, directory_path=args.save_dir)

    print('finished download')
    # api.download_all(df1.index,directory_path=args.save_dir)

if args.is_check2A:
    path_2A = args.save_dir.replace('/1C/', '/2A/').replace('/PRODUCT/', '')
    print(path_2A)

    file_existing_correct = args.save_dir + '/correct_zip.txt'
    assert os.path.isfile(file_existing_correct), 'checksum of 1C files first'
    # print('reading files in correct_zip.txt for 2A verification 2A')
    lines = [line.rstrip('\n')[:44] for line in open(file_existing_correct)]
    if '/1C/' in args.save_dir:
        titles_ = [x.replace('_MSIL1C_', '_MSIL2A_') for x in lines]
    else:
        # files are already 2A
        titles_ = lines
    # else:
    #     titles_ = [x.replace('_MSIL1C_','_MSIL2A_') for x in df.title.sort_values()]

    print(f'checking 2A product for {len(titles_)} correct zip files')
    file_existing_correct = path_2A + '/correct_2A.txt'
    if os.path.isfile(file_existing_correct):
        checked_ds = [line.rstrip('\n')[:44] for line in open(file_existing_correct)]

        titles_ = [x for x in titles_ if x not in checked_ds]
    titles_ = sorted(titles_)
    if len(titles_) > 0:
        print(f'checkcount for {len(titles_)} files pending')

        # existing_2A_titles = [x for x in titles_ if os.path.isdir(f'{path_2A}/{x}.SAFE')]
        
        # find the 2A file with any processing date
        existing_2A_titles = []
        for tile in titles_:
            existing_list = glob.glob(f'{path_2A}/{tile}*.SAFE',recursive=False)
            if len(existing_list) > 0:
                # assert len(existing_list) == 1, f'there should only be one 2A per 1C {existing_list}'
                title_ = os.path.basename(existing_list[0]).replace('.SAFE','')
                existing_2A_titles.append(title_)          

        # get_jp2_count = lambda x: len(x + '/**/*.jp2', recursive=True))
        get_jp2_count = lambda x: len(glob.glob(path_2A + '/' + x + '.SAFE/**/*.jp2', recursive=True))
        correct_2A_tiles = [x for x in existing_2A_titles if get_jp2_count(x) >= 40]

        # # add correct 2A in tar
        # tar_files = glob.glob(path_2A+'/*.tar')
        # for i in tar_files:
        #     tFile = tarfile.open(i,'r')
        #     existing_tar = [x for x in tFile.getnames() if x.endswith('SAFE')]
        #     correct_2A_tiles.append(existing_tar)

        # # add correct 2A in h5
        # h5_files = glob.glob(path_2A + '/*.he5')
        # for i in h5_files:
        #     h5File = h5py.File(i, 'r')
        #     existing_tar = list(h5File.keys())
        #     h5File.close()
        #     correct_2A_tiles.append(existing_tar)

        correct_2A_tiles = set(correct_2A_tiles)

        print(f'checkcount correct for additional {len(correct_2A_tiles)} tiles')
        with open(file_existing_correct, 'a+') as f:
            for item in correct_2A_tiles:
                f.write("%s\n" % item.split('/')[-1])

        if len(titles_) > len(correct_2A_tiles):
            if args.is_delete:
                raise NotImplementedError('pending fix titles with new proc date')
                missing_tiles = [x for x in titles_ if not x in correct_2A_tiles]
                print(f'deleting  {len(missing_tiles)} incorrect files an creating missing_2A.txt')
                file_missing = path_2A + '/missing_2A.txt'
                with open(file_missing, 'w') as f:
                    for item in missing_tiles:
                        f.write("%s\n" % item.split('/')[-1])
                        if args.is_delete:
                            shutil.rmtree(f'{path_2A}/{item}.SAFE', ignore_errors=True)
            else:
                print('creating incorrect_2A.txt and missing_2A.txt in separate files')
            incorrect_tiles = [x for x in existing_2A_titles if not x in correct_2A_tiles]
            file_incorrect = path_2A + '/incorrect_2A.txt'
            with open(file_incorrect, 'w') as f:
                for item in incorrect_tiles:
                    f.write("%s\n" % item.split('/')[-1])

            missing_tiles = [x for x in titles_ if not x in existing_2A_titles]
            file_missing = path_2A + '/missing_2A.txt'
            with open(file_missing, 'w') as f:
                for item in missing_tiles:
                    f.write("%s\n" % item.split('/')[-1])
        else:
            print('all 2A correct')
if args.is_cleanup:
    path_2A = args.save_dir.replace('/1C/', '/2A/').replace('/PRODUCT/', '')
    print(path_2A)
    file_existing_correct = path_2A + '/correct_2A.txt'

    correct_2A = [line.rstrip('\n') for line in open(file_existing_correct)]

    filelist_1C = glob.glob(args.save_dir + '/*_MSIL1C_*.SAFE')
    print('initial file list', len(filelist_1C))

    for file in filelist_1C:
        file_ = file.split('/')[-1].replace('_MSIL1C_', '_MSIL2A_').replace('.SAFE', '')
        if file_ in correct_2A:
            print('correct file found, removing SAFE in 1C')
            shutil.rmtree(file, ignore_errors=True)

    filelist_1C = glob.glob(args.save_dir + '/*_MSIL1C_*.SAFE')
    print('SAFE folders left ', len(filelist_1C))

if args.is_tar2A:
    path_2A = args.save_dir.replace('/1C/', '/2A/').replace('/PRODUCT/', '')

    file_existing_correct = path_2A + '/correct_2A.txt'
    correct_2A = [line.rstrip('\n') for line in open(file_existing_correct)]

    ds_2A = pd.DataFrame({'file': correct_2A})

    ds_2A['orbit_tile'] = ds_2A['file'].map(lambda x: '_'.join(x.split('_')[4:6]))
    ds_2A['path'] = ds_2A['file'].map(lambda x: f'{path_2A}/{x}.SAFE')
    ds_2A['correct'] = ds_2A['path'].map(lambda x: os.path.isdir(x))

    for i, group in ds_2A.groupby('orbit_tile'):
        print(i, group.shape)
        # Create Tar file
        tarname = f"{path_2A}/{i}.tar"
        tFile = tarfile.open(tarname, 'a')

        existing_tar = [x for x in tFile.getnames() if x.endswith('SAFE')]
        for f in group.path:
            get_jp2_count = lambda x: len(glob.glob(x + '/**/*.jp2', recursive=True))
            fname = f.split('/')[-1]
            if not fname in existing_tar:
                assert get_jp2_count(f) > 40, '2A file incorrect or deleted'
                tFile.add(f, arcname=fname)
                print('added', fname)

        print(f'created {tarname} with {group.shape} files')
        tFile.close()

        if args.is_delete:
            tFile = tarfile.open(tarname, 'r')

            existing_tar = [x for x in tFile.getnames() if x.endswith('SAFE')]
            for file in existing_tar:
                if os.path.isdir(path_2A + '/' + file):
                    shutil.rmtree(path_2A + '/' + file, ignore_errors=True)
                    print(f'deleted {file} after tar check')

    print(ds_2A)

#str_type = h5py.new_vlen(str)
# str_type = h5py.special_dtype(vlen=str)

def get_newname(fname):
    newname = fname.split('_', maxsplit=3)[-1].replace('.jp2', '')

    if '10m' in newname:
        prefix = '10m/'
    elif '20m' in newname:
        prefix = '20m/'
    elif '60m' in newname:
        prefix = '60m/'
    else:
        prefix = 'other/'
        if fname.startswith('T'):
            newname = "1C_" + newname
    return prefix + newname


if args.is_h52A:
    path_2A = args.save_dir.replace('/1C/', '/2A/').replace('/PRODUCT/', '')

    file_existing_correct = path_2A + '/correct_2A.txt'
    correct_2A = [line.rstrip('\n') for line in open(file_existing_correct)]

    ds_2A = pd.DataFrame({'file': correct_2A})

    ds_2A['orbit_tile'] = ds_2A['file'].map(lambda x: '_'.join(x.split('_')[4:6]))
    ds_2A['path'] = ds_2A['file'].map(lambda x: f'{path_2A}/{x}.SAFE')
    ds_2A['correct'] = ds_2A['path'].map(lambda x: os.path.isdir(x))

    for i, group in ds_2A.groupby('orbit_tile'):
        print(i, group.shape)
        # Create Tar file
        h5filename = f"{path_2A}/{i}.he5"
        ds_h5 = h5py.File(h5filename, 'a')


        def add_safe(safe):
            namegroup = safe.split('/')[-1]
            print('processing', namegroup)
            grp = ds_h5.create_group(namegroup)
            filelist = glob.glob(safe + "/**/*.jp2", recursive=True)
            for file in filelist:
                fname = file.split('/')[-1]
                newname = get_newname(fname)

                ds = gdal.Open(file)
                data = ds.ReadAsArray()
                data_h5 = grp.create_dataset(newname, data=data, dtype=data.dtype, compression="gzip",
                                             compression_opts=9)
                data_h5.attrs['Projection'] = ds.GetProjectionRef()
                data_h5.attrs['GeoTransform'] = ds.GetGeoTransform()
            filelist = glob.glob(safe + "/*.xml")
            for file in filelist:
                fname = file.split('/')[-1]
                data_h5 = grp.create_dataset('meta/' + fname, shape=(1,), dtype=str_type)
                xmldata = open(file, 'rb')
                data_h5[:] = xmldata.read()


        existing_h5 = ds_h5.keys()
        for f in group.path:
            get_jp2_count = lambda x: len(glob.glob(x + '/**/*.jp2', recursive=True))
            fname = f.split('/')[-1]
            if not fname in existing_h5:
                assert get_jp2_count(f) > 40, '2A file incorrect or deleted'
                add_safe(f)
                # tFile.add(f, arcname=fname)
                # print('added',fname)

        print(f'created {h5filename} with {group.shape} files')
        ds_h5.close()

        if args.is_delete:
            ds_h5 = h5py.File(h5filename, 'r')
            existing_h5 = ds_h5.keys()

            for file in existing_h5:
                if os.path.isdir(path_2A+'/'+file):
                    shutil.rmtree(path_2A+'/'+file,ignore_errors=True)
                    print(f'deleted {file} after tar check')

    print(ds_2A)
