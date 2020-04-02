---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3 (conda)
    language: python
    name: python3
---

```python
from osgeo import gdal, ogr, osr
import os
import glob

import numpy as np
```

```python

def convert_to_shp(points, is_overwrite=False):
    if points.endswith('.kml'):
        new_points=points.replace('.kml','.shp')
        if not os.path.exists(new_points) or is_overwrite:
            srcDS = gdal.OpenEx(points)
            ds = gdal.VectorTranslate(new_points, srcDS, format='ESRI Shapefile')
            ds = None
            points = new_points
    return points 


def keep_name_only(points):
    dataSource = ogr.Open(points, 1) 

    layer = dataSource.GetLayer()

    lyrdf = layer.GetLayerDefn()

    id_Name = lyrdf.GetFieldIndex('Name')
    attr_N = lyrdf.GetFieldCount()
    print(attr_N, id_Name)
    for i in range(attr_N):
        if not i == id_Name:
            layer.DeleteField(i)   
    attr_N = lyrdf.GetFieldCount()
    print(attr_N)
    dataSource = None

    
def get_name_wkt(file):
    data = ogr.Open(file)
    
    out= {}
    for layer in data:
        layer.ResetReading()
        for feature in layer:
            items_ = feature.items()
            geom = feature.geometry()
            out[items_['Name']] = geom.ExportToWkt()
#             print(geom)
#             print('Feature Geometry:', feature.geometry())
    return out
```

```python
tile='*'
# PATH='/scratch/andresro/leon_igp'
PATH='/home/pf/pfstaff/projects/andresro'

filelist = glob.glob(PATH+'/barry_palm/data/labels/palm_annotations/{}/**/*.kml'.format(tile))

dirnames = list({os.path.dirname(x) for x in filelist})
print(len(dirnames))
```

```python jupyter={"outputs_hidden": true}
for file in filelist:
    file1 = convert_to_shp(file, is_overwrite=False)
    keep_name_only(file1)
```

```python
def get_featname(file, is_assert=False):
    data = ogr.Open(file)
    out = []
    for layer in data:
        layer.ResetReading()
        for feature in layer:
            items_ = feature.items()
            geom = feature.geometry()
            geomtype = geom.GetGeometryName()
            out.append({'Name':items_['Name'].lower(),
                        'geom':geomtype,
                       'file':file})
    if is_assert:
        geom_types = set([x['geom'] for x in out])
        assert len(geom_types) <=1,f'{file} has more than one geom type: {geom_types}'
        pos_names = set([x for x in out if 'pos' in x['Name'] and 'GEOM' in x['geom']])
        neg_names = set([x for x in out if 'neg' in x['Name'] and 'GEOM' in x['geom']])
        assert len(pos_names)+len(neg_names) <=1,f'{file} has both positive and negative geometries'
    return out

```

```python
folder='/scratch/andresro/leon_work/barry_palm/data/labels/palm_annotations/T50NQL/group1'


filelist = glob.glob(folder+'/*.shp')

featnames = [get_featname(file, is_assert=True)[0] for file in filelist] # TODO fix if there is more than 1 feature in .shp
# pos_shp = [x for x,names in zip(filelist,featnames) if 'pos' in names[0] and 'POLYGON' in names[1]]
```

```python
poly_feat = ['POLY' in x[1] for x in featnames[2]]

```

```python

featnames
```

```python
set([x['geom'] for x in featnames[6]])
```

```python

pos_names = set([x for x in featnames[7] if 'pos' in x['Name'] and 'GEOM' in x['geom']])
neg_names = set([x for x in featnames[7] if 'neg' in x['Name'] and 'GEOM' in x['geom']])

```

```python
pos_names,neg_names
```

```python

```
