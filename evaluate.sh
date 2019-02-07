#!/usr/bin/env bash

## USAGE: bash evaluate.sh 0 coco1 # To run coco1 config with GPU_ID = 0
set -x
set -e

GPU_ID=$1
DATASET=$2
#EXTRA_ARGS=${3:-}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}


case ${DATASET} in
  coco1)
    OBJECT='coco'
    WEIGHTS='/home/pf/pfstaff/projects/andresro/sparse/training/snapshots/model-1_size-32_scale-10_nchan12temp6Feb'
    MODEL='1'
    IMG='600,300'
    XY_CORNER="20,20"
    SAVE_DIR='/home/pf/pfstaff/projects/andresro/sparse/evaluation'
    EXTRA_ARGS=$EXTRA_ARGS'short'
    ;;
  coco2)
    OBJECT='coco'
    WEIGHTS='/scratch/andresro/leon_work/sparse/training/snapshots/coco/model-1a_size-32_scale-2_nchan12'
    MODEL='1a'
    IMG='600,300'
    XY_CORNER="20,20"
    SAVE_DIR='/home/pf/pfstaff/projects/andresro/sparse/evaluation'
    EXTRA_ARGS=$EXTRA_ARGS'short'
    ;;
  coco3)
    OBJECT='coco'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/coco_2017/raw_scale10.0/117.81-8.82-117.84-8.88'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/coco/checkpoints-128/CloudFree-0.9DL2_{version}/model.ckpt-104000'
    MODEL='DL2'
    IMG='600,300'
    XY_CORNER="20,20"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  coco4)
    OBJECT='coco'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/coco_2017/raw_scale10.0/117.81-8.82-117.84-8.88'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/coco/checkpoints-128/CloudFree-0.9DL3_{version}/model.ckpt-104000'
    MODEL='DL3'
    IMG='600,300'
    XY_CORNER="20,20"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  coco5)
    OBJECT='coco'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/coco_2017/raw_scale10.0/117.81-8.82-117.84-8.88'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/coco/checkpoints-128/Simple_30deep/model.ckpt-20000'
    MODEL='Simple_5'
    IMG='600,300'
    XY_CORNER="20,20"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    EXTRA_ARGS=$EXTRA_ARGS'short'
    ;;
  coco1b)
    OBJECT='coco'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/coco_2017/raw_scale10.0/117.81-8.82-117.84-8.88'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/coco/checkpoints-128/CloudFree-0.9Simple_atrous_{version}/model.ckpt-84000'
    MODEL='Simple_atrous'
    IMG='650,330'
    XY_CORNER="0,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  coco2b)
    OBJECT='coco'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/coco_2017/raw_scale10.0/117.81-8.82-117.84-8.88'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/coco/checkpoints-128/CloudFree-0.9Simple_{version}/model.ckpt-104000'
    MODEL='Simple'
    IMG='650,300'
    XY_CORNER="0,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  coco2c)
    OBJECT='coco'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/coco_2017/raw/117.81-8.82-117.84-8.88'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/coco/checkpoints-128/cl1_free0.9Simple_7scale1/model.ckpt-70000'
    MODEL='Simple_7'
    IMG='650,300'
    XY_CORNER="0,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    EXTRA_ARGS=$EXTRA_ARGS'scale1'

    ;;
  palm1)
    OBJECT='palm'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/palm_2017a/raw_scale10.0/101.45-0.53-101.62-0.55'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/palm/checkpoints-128/cl1_free0.9Simple_atrous/model.ckpt-60000'
    MODEL='Simple_atrous'
    IMG='200,1100'
    XY_CORNER="22,500"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  palm2)
    OBJECT='palm'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/palm_2017a/raw_scale10.0/101.45-0.53-101.62-0.55'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/palm/checkpoints-128/cl1_free0.9Simple/model.ckpt-86000'
    MODEL='Simple'
    IMG='200,1100'
    XY_CORNER="22,500"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  palm3)
    OBJECT='palm'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/palm_2017a/raw_scale10.0/101.45-0.53-101.62-0.55'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/palm/checkpoints-128/cl1_free0.9DL2/model.ckpt-86000'
    MODEL='DL2'
    IMG='200,1100'
    XY_CORNER="22,500"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  palm4)
    OBJECT='palm'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/palm_2017a/raw_scale10.0/101.45-0.53-101.62-0.55'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/palm/checkpoints-128/cl1_free0.9DL3/model.ckpt-56000'
    MODEL='DL3'
    IMG='200,1100'
    XY_CORNER="22,500"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  olives1)
    OBJECT='olives'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/olives_2016/raw_scale10.0/-3.79-37.78--3.77-37.9'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/olives/checkpoints-128/cl1_free0.9Simple_atrousshort/model.ckpt-5000'
    MODEL='Simple_atrous'
    IMG='470,180'
    XY_CORNER="5,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  olives2)
    OBJECT='olives'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/olives_2016/raw_scale10.0/-3.79-37.78--3.77-37.9'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/olives/checkpoints-128/cl1_free0.9Simpleshort/model.ckpt-5000'
    MODEL='Simple'
    IMG='470,180'
    XY_CORNER="5,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  olives3)
    OBJECT='olives'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/olives_2016/raw_scale10.0/-3.79-37.78--3.77-37.9'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/olives/checkpoints-128/cl1_free0.9DL2short/model.ckpt-5000'
    MODEL='DL2'
    IMG='470,180'
    XY_CORNER="5,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  olives4)
    OBJECT='olives'
    DATA='/home/pf/pfstaff/projects/barry_palm/data/3A/olives_2016/raw_scale10.0/-3.79-37.78--3.77-37.9'
    WEIGHTS='/home/pf/pfstaff/projects/barry_palm/training_leon/snapshots/olives/checkpoints-128/cl1_free0.9DL3short/model.ckpt-5000'
    MODEL='DL3'
    IMG='470,180'
    XY_CORNER="5,0"
    SAVE_DIR='/home/pf/pfstaff/projects/barry_palm/evaluation'
    ;;
  *)
    echo "Option not defined"
    exit
    ;;
esac



PATH_OUTPUT=$HOME/palm/output
SAVE_DIR_OBJECT=${SAVE_DIR}/${OBJECT}/${MODEL}_${EXTRA_ARGS}

LOG="${PATH_OUTPUT}/inference_${DATASET}_`date +'%Y-%m-%d_%H-%M-%S'`.txt"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# echo $search_path
CUDA_VISIBLE_DEVICES=${GPU_ID} time python2 evaluate_single.py \
    ${WEIGHTS} \
    --model=$MODEL \
    --img-size=$IMG \
    --save-dir=$SAVE_DIR_OBJECT \
    --xy-corner=$XY_CORNER \
    --pred-range=0,4
    #${EXTRA_ARGS}


