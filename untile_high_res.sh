#!/usr/bin/env bash

DATASET=$1

case ${DATASET} in
  coco)
    OBJECT='coco'
    DS='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170215T022751_N0204_R046_T50PNQ_20170215T023409.SAFE/MTD_MSIL2A.xml'
    ROI='117.81,8.82,117.92,8.9'
    ;;
  palm)
    OBJECT='palm'
    DS='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/palm_2017/S2A_MSIL2A_20170104T033122_N0204_R018_T47NQA_20170104T033834.SAFE/MTD_MSIL2A.xml'
    ROI='101.45,0.55,101.62,0.48'
    ;;
  *)
    echo "Option not defined"
    exit
    ;;
esac
set -x

SAVEDIR=/home/pf/pfstaff/projects/andresro/sparse/data/$OBJECT
IMGDIR=/home/pf/pfstaff/projects/andresro/barry_palm/obj_det/${OBJECT}/DB_JPEG/aerial_images_tiles
#GSD='0.5'
for GSD in 1 2.5 5
do
python -u ~/code/supertime/untile_images.py --ref-dataset=$DS --roi_lon_lat=$ROI --save-dir=$SAVEDIR --image-dir=$IMGDIR --is-merge --gsd=$GSD
done
#--is-debug


