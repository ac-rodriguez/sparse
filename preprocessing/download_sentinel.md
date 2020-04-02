---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3.6.9 usr/bin
    language: python
    name: python36964bit4de6fba84f53437fbed552466d2e9e78
---

## Downloading Sentinel-2 imagery

This script can be used to download all sentinel-2 images inside a geographical area defined by a `.shp` file and certain time-range. For large areas, this process will take a long time since the bandwitdh for downloading from ESA is not too large, you can use this script to query only the tiles to be downloaded and save it as a `.pkl` file. Then you can use `download1C_df.sh` to load the `.pkl` file and download then.


```python
from osgeo import gdal, osr, ogr, gdalconst
import os
import numpy as np
from shapely.geometry import mapping, shape
from shapely.wkt import loads
from shapely.geometry import Polygon
import json
import xml.etree.ElementTree as ET 
import glob
%matplotlib inline
import pandas as pd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date

```

## Set parameters


Download the .shp files from [here](https://www.diva-gis.org/gdata) for each country of interest


## Load the country shape file

```python
#root_dir = '/scratch/andresro/leon_igp/barry_palm/data'
root_dir = '/home/pf/pfstaff/projects/andresro/barry_palm/data'

country='Malaysia'

if country == 'Phillipines':
    loc='asia_2019'
    # loc='phillipines_2017'
    adm1_path = '/home/pf/pfstaff/projects/andresro/data/countries/phillipines/PHL_adm1.shp'
    NAME = 'all'
#     NAME = ['Palawan','Cebu','Davao del Norte','Davao del Sur','Davao Oriental','Batangas','Quezon','Rizal','Laguna']
elif country == 'Malaysia':
    loc='asia_2019'
#     loc='palmcountries_2017'
    NAME = 'all'
    adm1_path = '/home/pf/pfstaff/projects/andresro/data/countries/malaysia/MYS_adm1.shp'
#     NAME=['Sarawak']
elif country == 'Indonesia':
    loc='asia_2019'
#     loc='palmcountries_2017'
    adm1_path = '/home/pf/pfstaff/projects/andresro/data/countries/indonesia/IDN_adm1.shp'
    NAME = 'all'
#     NAME = ['Kalimantan Barat','Riau','Sulawesi Barat','Sulawesi Tengah','Sulawesi Utara','Gorontalo']


product_dir = os.path.join(root_dir,'1C',loc,'PRODUCT')
save_dir = os.path.join(root_dir,'1C','dataframes_download')
save_dir1 = os.path.join(save_dir,loc)
if not os.path.exists(save_dir1):
    os.makedirs(save_dir1)
    
if not os.path.exists(product_dir):
    os.makedirs(product_dir)
    
print('product_dir: ', product_dir)

```

```python

fieldname = 'NAME_1'

shp = ogr.Open(adm1_path)
lyr = shp.GetLayer(0)
lyrdf =lyr.GetLayerDefn()
id_ = lyrdf.GetFieldIndex(fieldname)
    
print('Total features', lyr.GetFeatureCount())
features_extent = {}
features_polygones = {}
for i in range(lyr.GetFeatureCount()):
    feat = lyr.GetFeature(i)
    value =feat.GetField(id_)
#     if value == name_:
    geom=feat.GetGeometryRef()
    extent = geom.GetEnvelope()
    lon1,lat1 = extent[0],extent[2]
    lon2,lat2 = extent[1],extent[3]
    wkt_ext = f'POLYGON(({lon1} {lat1}, {lon1} {lat2}, {lon2} {lat2},  {lon2} {lat1},  {lon1} {lat1} ))'
    features_extent[value] = wkt_ext
    features_polygones[value]=loads(geom.ExportToWkt())

    
```

## Get Sentinel-2 tile names in Polygon

You can download the sentinel-2 tiles from [here](https://sentinel.esa.int/web/sentinel/missions/sentinel-2/data-products) as .xml and convert them as .shp with QGIS to create ´Features.shp´

```python
sentinel2_tiles_path = '/home/pf/pfstaff/projects/nlang_HCS/data/Sentinel2_mission/sentinel2_tiles/Features.shp'
driver = ogr.GetDriverByName('ESRI Shapefile')
sentinel2_tiles = driver.Open(sentinel2_tiles_path, 0) # 0 means read-only. 1 means writeable.

print('Opened {}'.format(sentinel2_tiles_path))
layer = sentinel2_tiles.GetLayer()
featureCount = layer.GetFeatureCount()
print('Number of layers: ', sentinel2_tiles.GetLayerCount())
# print(layer.GetLayerDefn())
print("Number of features: ", featureCount)


def getGeom(Shapefile, shapely = True):
    feature_dict={}
    n_layers = Shapefile.GetLayerCount()
    wkt_list  = []
    for _ in range(n_layers):
        Shapefile_layer = Shapefile.GetLayer()

        n_points = Shapefile_layer.GetFeatureCount()

        for _ in range(n_points):
            feat = Shapefile_layer.GetNextFeature()
            if feat:
                name = feat.GetFieldAsString("Name")
                geom = feat.geometry().ExportToWkt()
                if shapely:
                    geom = loads(geom)
                wkt_list.append(geom)
                # save in dictionary
                feature_dict[name]=geom

    print('{} geometries loaded'.format(len(wkt_list)))

    return wkt_list, feature_dict
    
tiles_geometry, feature_dict = getGeom(Shapefile=sentinel2_tiles)

```

```python

roi_tiles_per_feature = {}

for name, poly in features_polygones.items():

    for tile, tile_poly in feature_dict.items():
        if poly.intersects(tile_poly):
            if name in roi_tiles_per_feature.keys():                
                roi_tiles_per_feature[name].append(tile)
            else:
                roi_tiles_per_feature[name] = [tile]
#     print(name,len(roi_tiles_per_feature[name]))

```

```python
df = pd.DataFrame.from_dict(roi_tiles_per_feature, orient='index')

df['count'] = df.shape[1]-df.isnull().sum(axis=1)
df = df.sort_values('count',ascending=False)
df['count'].sum()
# for key, value in roi_tiles_per_feature.items():
#     print(key,len(value))
```

```python
a = set(np.array(df.drop('count', axis=1)).flatten())
len(a)
df['tiles_country'] = len(a)
```

```python
df1 = []
for key, val in roi_tiles_per_feature.items():
    for tile in val:
        df1.append([key,tile])
df1 = pd.DataFrame(columns=['Region','Tile'], data=df1)
```

```python
set_tiles = {tile for _,val in roi_tiles_per_feature.items() for tile in val}

df1 = []
for tile in set_tiles:
    keys = [key for key, val in roi_tiles_per_feature.items() if tile in val]
    df1.append([', '.join(keys),tile])
df1 = pd.DataFrame(columns=['Region','Tile'], data=df1)

```

```python
filename = save_dir1+'/'+f'Tiles_per_region_{country}_1.csv'
# df1.to_csv(filename)
# print(filename, 'saved!')
```

```python
df_tiles = df
df_tiles.index
```

## Query tiles from SCIHUB server

If you do not have a scihub account create one [here](https://scihub.copernicus.eu/dhus/#/self-registration).

Before downloading we will just query all the sentinel tiles to create a database where we can track which tiles are available to download.

Now you can add your details to download the corresponding images. You can choose which images to download depending on the state name or 'all' for downloading all states inside the .shp file

```python
# connect to the API

username ='andrescamilor'
password='ichLVub40'
api = SentinelAPI(username, password, 'https://scihub.copernicus.eu/dhus')

```

```python
is_all = True
is_load =True

if NAME == 'all':
    is_all = True
    
if is_load:
    name_ = '_'.join(NAME).replace(' ','_') if not is_all else 'all'
    file_ = glob.glob(f'{save_dir1}/{country}_{name_}_*.pkl')[-1]
    
    df_download = pd.read_pickle(file_)
else:
    
    products_df = []
    # search by polygon, time, and SciHub query keywords
    NAME1 = set(df_tiles.index) if is_all else NAME
    for name_ in NAME1:
        print(name_)
        products = api.query(area=features_extent[name_],
                             # CHANGE desired time-frame here
                             date=('20190101', date(2019, 12,31)),
                             processinglevel = 'Level-2A',
                             platformname='Sentinel-2')

        # convert to Pandas DataFrame
        products_df.append(api.to_dataframe(products))



    df_out = None
    for d_ in products_df:
        if df_out is None:
            df_out = d_
        else:
            df_out = df_out.append(d_)

    # df_out.shape

    df1 = df_out.copy()
    df1.drop_duplicates(subset=['title'], inplace =True)
#     df1[df1.tileid.isna()].shape, df1[df1.relativeorbitnumber.isna()].shape

    df1['sizeMB'] = df1['size'].map(lambda x: float(x.replace(' MB','')) if 'MB' in x else 1000*float(x.replace(' GB','')))
    df1['tileid'] = df1['title'].map(lambda x: x.split('_')[5][1:])


#     Remove Tiles not in polygon

    index = None
    for name_ in NAME1:
        index_ = [x in roi_tiles_per_feature[name_] for x in df1.tileid]
        if index is None:
            index = index_
        else:
            index = np.logical_or(index_,index)


    # index
    df1 = df1[index]

    df_download = df1.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True]).groupby(['tileid','relativeorbitnumber']).head(10)
    df_download = df_download.sort_values(['tileid','relativeorbitnumber','cloudcoverpercentage'])


    print(len(np.unique(df_download.tileid)),len(np.unique(df_download.title)))

    for counter, (id_, d) in enumerate(df_download.groupby(['tileid','relativeorbitnumber'])):
        print(counter, id_,f' N {d.shape[0]} mean cc {d.cloudcoverpercentage.mean():.2f}')

    df_download.sizeMB.sum()/(10*60*60)
    name_ = '_'.join(NAME).replace(' ','_') if not is_all else 'all'
#     name_ = '_'.join(NAME).replace(' ','_')
    file_=f'{save_dir1}/{country}_{name_}_{df_download.shape[0]}.pkl'
    
    df_download.to_pickle(file_)
```

```python
df_download.shape
```

```python
file1_ = file_.replace('.pkl','.txt')

with open(file1_, 'w') as f:
    for item in df_download.title:
        f.write("%s\n" % item)
print(file1_,'saved')        
```

```python
df_download.title[0]
```

```python
print('total size',np.sum(df_download.sizeMB))
```

This script could be used to download directly all the scrips but if we have too many tiles, this will usually take several days to complete.

```python
# # download sorted and reduced products
# api.download_all(df.index,directory_path=product_dir)
```

```python
df_download.groupby('processinglevel').count()
```

## Download status and location

```python
save_dir = os.path.join(root_dir,'ref_dataframes')

dirs_ = [root_dir+'/1C/dataframes_download/palmcountries_2017/Malaysia_all_1150.pkl',
         root_dir+'/1C/dataframes_download/palmcountries_2017/Indonesia_all_8410.pkl',
         root_dir+'/1C/dataframes_download/phillipines_2017/Phillipines_all_1840.pkl',
         root_dir+'/1C/dataframes_download/asia_2019/Phillipines_all_1835.pkl',
         root_dir+'/1C/dataframes_download/asia_2019/Malaysia_all_1152.pkl',
        ]
df_ = [pd.read_pickle(dir_) for dir_ in dirs_]
df_ = pd.concat(df_)

df_['1C_path'] = None
df_['2A_path'] = None
df_['correct2A'] = None



```

```python
base_path =f'{root_dir}/1C/*/PRODUCT/'
# base_path ='/home/pf/pfstaff/projects/andresro/barry_palm/data/1C/palm_2017/PRODUCT/'
print(base_path)
filelist = glob.glob(base_path+'*.zip')

titlelist = [os.path.split(x)[-1].replace('.zip','') for x in filelist]
def path_if_exists(x):
    if x['1C_path'] is None:
        if x['title'] in titlelist:
            return filelist[titlelist.index(x['title'])]
    return x['1C_path']

df_['1C_path'] = df_.apply(path_if_exists,axis=1)


base_path =f'{root_dir}/2A/*/'
# base_path ='/home/pf/pfstaff/projects/andresro/barry_palm/data/1C/palm_2017/PRODUCT/'
print(base_path)
filelist = glob.glob(base_path+'*.SAFE')

titlelist = [os.path.split(x)[-1].replace('.SAFE','') for x in filelist]
def path_if_exists(x):
    if x['2A_path'] is None:
        title_ = x['title'].replace('_MSIL1C_','_MSIL2A_')
        if title_ in titlelist:
            return filelist[titlelist.index(title_)]
    return x['2A_path']

df_['2A_path'] = df_.apply(path_if_exists,axis=1)

```

```python
def check2A(x):
    if x['correct2A'] is None and x['2A_path'] is not None:
        file_ = x['2A_path']+'/jp2count.txt'
        if not os.path.isfile(file_):
            jp2count = len(glob.glob(x['2A_path']+'/**/*.jp2', recursive=True))
            f = open(file_, "w")
            f.write(str(jp2count))
            f.close()
        else:
            f = open(file_, "r")
            jp2count = int(f.read())
            f.close()
        return jp2count
    return x['correct2A']
    
df_['correct_2A'] = df_.apply(check2A,axis=1)
```

```python
print(df_.shape)
'1Czip: ',np.sum(~df_['1C_path'].isna()), '2A:', np.sum(~df_['2A_path'].isna()), 'correct2A:',np.sum(df_['correct_2A'].dropna() >= 40)
```

```python
# SAVE DF
df_.to_pickle(root_dir+'/filestatus.pkl')
print(root_dir+'/filestatus.pkl','saved!')
```

## Check 1C downloads

```python
tiles = df_download.title.map(lambda x: '_'.join(x.split('_')[4:6]))
```

```python

base_path =root_dir+'/1C/{}/PRODUCT/'.format(loc) 
# base_path ='/home/pf/pfstaff/projects/andresro/barry_palm/data/1C/palm_2017/PRODUCT/'

filelist = glob.glob(base_path+'*.zip')

existing_ds = [os.path.split(x)[-1].replace('.zip','') for x in filelist]
# pending_ds = [x for x in df_download.title if x not in existing_ds]
pending_ds = [x not in existing_ds for x in df_download.title]

print('total',df_download.shape[0])
print(f'existing {len(existing_ds)} in {base_path}')
print('pending',np.sum(pending_ds))


```

```python
df_existing = df_download[~np.array(pending_ds)]

file_existing_correct = base_path+'/correct_zip.txt'
lines = [line.rstrip('\n') for line in open(file_existing_correct)]

is_checked = [x not in lines for x in df_existing.title]
df_to_check = df_existing[is_checked]

print('1C ds pending checksum:',df_to_check.shape[0])
```

```python
for id_, row in df_download.head(5).iterrows():
    print(row.link)
#     print(f'wget --content-disposition --continue --user={username} --password={password} "https://scihub.copernicus.eu/dhus/odata/v1/Products(\'{id_}\')/\$value" -P {save_dir}')
#     print(f'wget --content-disposition --continue --user={username} --password={password} "{row.link}"')
```

```python
# df_download.inde
base_path
```

## Check 2A files

```python
path=root_dir+'/1C/'+loc+'/PRODUCT/correct_zip.txt'
lines1C = [line.rstrip('\n') for line in open(path)]
lines1C = [x for x in lines1C if '2017' in x]
print('1C: ',len(lines1C))


path=root_dir+'/2A/'+loc+'/correct_2A.txt'
lines2A = [line.rstrip('\n') for line in open(path)]
lines2A = [x for x in lines2A if '2017' in x]
print('2A: ',len(lines2A))
```

```python
ds1C = pd.DataFrame({'title1C': lines1C})

ds1C['tile'] = ds1C.title1C.map(lambda x: x.split('_')[5])
ds1C['orbit'] = ds1C.title1C.map(lambda x: x.split('_')[4])

ds1C.head()

counts1C = ds1C.groupby(['tile','orbit']).count().rename({'title1C':'count1C'},axis=1)

ds1C = ds1C.set_index(['tile','orbit']).join(counts1C)
ds1C['title2A'] = ds1C.title1C.map(lambda x: x.replace('MSIL1C','MSIL2A'))
ds1C.head()
```

```python
ds2A = pd.DataFrame({'title2A': lines2A})

ds2A['tile'] = ds2A.title2A.map(lambda x: x.split('_')[5])
ds2A['orbit'] = ds2A.title2A.map(lambda x: x.split('_')[4])

ds2A.head()

counts2A = ds2A.groupby(['tile','orbit']).count().rename({'title2A':'count2A'},axis=1)
ds2A = ds2A.set_index(['tile','orbit']).join(counts2A)
ds2A['correct2A'] = True

ds2A.head()
```

```python
dsAll = ds1C.set_index('title2A').join(ds2A.set_index('title2A'))

#dsAll[dsAll.correct2A != True].title1C
#dsAll[dsAll.title1C]
dsAll.reset_index().columns 
```

```python
dsAll[dsAll.correct2A != True].head()
```

```python
# ds2A.groupby(['tile']).count()

counts2A.sort_index().sort_values(by='count2A',ascending=True)
```

```python
path='/scratch/andresro/leon_igp/barry_palm/data/2A/phillipines_2017/correct_2A.txt'
lines = [line.rstrip('\n') for line in open(path)]
```

```python

len(lines),len(set(lines))
```
