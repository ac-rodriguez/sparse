#!bin/env bash

## USAGE: bash evaluate.sh 0 coco1 # To run coco1 config with GPU_ID = 0
set -x
set -e

DATASET=$1

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:1:$len}

GSD=5.0


case ${DATASET} in
  coco)
    OBJECT='coco'
    LRFILE='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/coco_2017p/S2A_MSIL2A_20170205T022901_N0204_R046_T50PNQ_20170205T024158.SAFE/MTD_MSIL2A.xml'
    POINTSFILE='/home/pf/pfstaff/projects/andresro/barry_palm/data/labels/coco/points_detections.kml'
    ROI_TR='117.84,8.82,117.92,8.9'
    ROI_TR_LB='117.8821,8.87414,117.891,8.8654'
    ROI_VAL='117.81,8.82,117.84,8.88'
    ROI_VAL_LB='117.81,8.82,117.84,8.88'
    ;;
  palm)
    OBJECT='palm'
    LRFILE='/home/pf/pfstaff/projects/andresro/barry_palm/data/2A/palm_2017a/S2A_MSIL2A_20170921T032531_N0205_R018_T47NQA_20170921T034446.SAFE/MTD_MSIL2A.xml'
    POINTSFILE='/home/pf/pfstaff/projects/andresro/barry_palm/obj_det/palm/detections_inference/default/kml_geoproposals'
    ROI_TR='101.45,0.48,101.62,0.53'
    ROI_TR_LB='101.545,0.512,101.553,0.516'
    ROI_VAL='101.45,0.53,101.62,0.55'
    ROI_VAL_LB='101.45,0.53,101.62,0.55'
    ;;
  *)
    echo "Option not defined"
    exit
    ;;
esac



SCALE=$(bc <<< "10/$GSD")
HR=$PF/sparse/data/${OBJECT}/3000_gsd${GSD}.tif


SAVE_DIR=/home/pf/pfstaff/projects/andresro/sparse/training/snapshots

PATH_OUTPUT=$HOME/code/output_sparse/
SAVE_DIR_OBJECT=${SAVE_DIR}/${OBJECT}

LOG="${PATH_OUTPUT}/inference_${DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# echo $search_path
python2 -u main.py \
    --LR_file=$LRFILE \
    --HR_file=$HR \
    --points=$POINTSFILE \
    --roi_lon_lat_tr=$ROI_TR \
    --roi_lon_lat_tr_lb=$ROI_TR_LB \
    --roi_lon_lat_val=$ROI_VAL \
    --roi_lon_lat_val_lb=$ROI_VAL_LB \
    --save-dir=$SAVE_DIR_OBJECT \
    --scale=2 \
    --is-bilinear \
    --optimizer=adam \
    --lr=2.5e-4 \
    --lambda-weights=0 \
    --batch-size=8 \
    --patch-size=32 \
    --patch-size-eval=64 \
    ${EXTRA_ARGS}


