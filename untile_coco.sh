#!/usr/bin/env bash


DS='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170215T022751_N0204_R046_T50PNQ_20170215T023409.SAFE/MTD_MSIL2A.xml'

ROI='117.84,8.82,117.92,8.9'

SAVEDIR='/home/pf/pfstaff/projects/andresro/sparse/data'

IMGDIR='/home/pf/pfstaff/projects/andresro/barry_palm/obj_det/coco/DB_JPEG/aerial_images_tiles'

GSD='1.0'
set -x

python -u ~/code/supertime/untile_images.py --ref-dataset=$DS --roi_lon_lat=$ROI --save-dir=$SAVEDIR --image-dir=$IMGDIR --is-merge --gsd=$GSD
#--is-debug


