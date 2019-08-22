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
  olives)
    OBJECT='olives'
    DS='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/olives_2016/S2A_USER_PRD_MSIL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.SAFE/S2A_USER_MTD_SAFL2A_PDMC_20160614T005258_R094_V20160613T110559_20160613T110559.xml'
    ROI='-3.9,37.78,-3.77,37.9'
    ;;
  cars)
    OBJECT='cars'
    DS='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/cars_2017/S2A_MSIL2A_20171230T183751_N0206_R027_T11SMU_20171230T202151.SAFE/MTD_MSIL2A.xml'
    ROI='-117.401,34.594,-117.39,34.581'
    ;;
  vaihingen)
    OBJECT='vaihingen'
    DS='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/cars_2017/S2A_MSIL2A_20171230T183751_N0206_R027_T11SMU_20171230T202151.SAFE/MTD_MSIL2A.xml'
    ROI='-117.401,34.594,-117.39,34.581'
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
for GSD in 0.3125 0.625 1.25 1 2.5 5
do
python -u ~/code/supertime/untile_images.py \
    --ref-dataset=$DS \
    --roi_lon_lat=$ROI \
    --save-dir=$SAVEDIR \
    --image-dir=$IMGDIR \
    --is-merge \
    --gsd=$GSD
done
#--is-debug


