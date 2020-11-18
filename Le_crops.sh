#!/bin/bash

set -x
set -e

models=(
        #simpleA30
        simpleA9 # simpleA20
        )
# kernels=(2)
#datasets=(palmcocotiles2_coco_palm_kalim) # palmcocotiles2_coco2_palm2)
#datasets=(palmpeninsulanew palmsarawak3 palmsabah) #palmpeninsulanew1 palmpeninsulanew2)
#datasets=(palmriau palmriau1)
#datasets=(palmsabah palmsarawak3 palmpeninsulanew palmriau) 
#datasets=(palmsarawak1 palmsarawak2) #abah palmsabah1) #palmsarawak palmsarawak1 palmsarawak2 palmsarawak3 palmsarawaksabah)
#datasets=(palmborneo)
datasets=(
        # palm4748a
        # palm3
        palm4_act
        # palm4
        #cocopreactive
        )
#datasets=(cocopalawanplus cocopalawanplus1)

# lambdasreg=(0.5)
others=( 
        # "--is-use-location --fusion-type=soft"
        # "--is-use-location --fusion-type=soft"
        # "--is-use-location --fusion-type=soft"
        # "--is-use-location --fusion-type=soft"
        # "--is-use-location --fusion-type=soft"
        ""
        # "--is-use-location --fusion-type=soft --is-dropout-uncertainty"
        # "--is-dropout-uncertainty"
        # "--is-resume"
        #"--is-use-location --fusion-type=soft"
        # "--active-samples=5"
        # "--active-samples=10"
        # "--active-samples=15"
        # "--active-samples=30"
        # "--active-samples=50"        
        # "" "" "" "" ""
        #"--is-dropout-uncertainty --is-val --n-eval-dropout=1"
        #"--is-dropout-uncertainty --is-val --n-eval-dropout=5"
        #"--is-dropout-uncertainty --is-val --n-eval-dropout=30"
        ) #1 0.001) # 0.1 0.01 0.001)
# optimizers=(adam)
# lrates=(2.5e-4) # 1e-5)

others1=(
        # "" "" "" "" ""
        # "--tag=a" "--tag=b" "--tag=c" "--tag=d"
        # "--tag=11septa"
        # "--tag=11sept" "--tag=11septa" "--tag=11septb" "--tag=11septc" "--tag=11septd"
       # '--patch-size=16 --batch-size=128'
#        '--patch-size=32 --batch-size=32'
        #'--patch-size=64 --batch-size=16'
        # "--rand-option=a"
        # "--rand-option=b" "--rand-option=b"
        # "--rand-option=c" "--rand-option=c"
        # "--rand-option=d"
        # "--rand-option=e" "--rand-option=e"
        # "--rand-option=ar" "--rand-option=ar"
        # "--rand-option=br" "--rand-option=br" "--rand-option=br"
        # "--rand-option=cr" "--rand-option=cr" "--rand-option=cr"
        # "--rand-option=dr" "--rand-option=dr" "--rand-option=dr"
        # "--rand-option=er" "--rand-option=er"
        # "--rand-option=ar"
        # "--rand-option=br"
        # "--rand-option=cr"
        # "--rand-option=dr"
        # "--rand-option=er"
        # "--rand-option=opt"
        # "--rand-option=ard"
        # "--rand-option=brd"
        # "--rand-option=crd"
        # "--rand-option=drd"
        # "--rand-option=erd"
        # "--rand-option=opt"
        # "--rand-option=optt" "--rand-option=optt" "--rand-option=optt" "--rand-option=optt"
        # "--dataset palm4748a_50ard"
        # "--dataset palm4748a_50ard --tag a"
        # "--dataset palm4748a_50ard --tag b"
        # "--dataset palm4748a_50ard --tag c"
        # "--dataset palm4748a_50brd"
        # "--dataset palm4748a_50brd --tag a"
        # "--dataset palm4748a_50brd --tag b"
        # "--dataset palm4748a_50brd --tag c"
        # "--dataset palm4748a_50crd"
        # "--dataset palm4748a_50crd --tag a"
        # "--dataset palm4748a_50crd --tag b"
        # "--dataset palm4748a_50crd --tag c"
        # "--dataset palm4748a_50drd"
        # "--dataset palm4748a_50drd --tag a"
        # "--dataset palm4748a_50drd --tag b"
        # "--dataset palm4748a_50drd --tag c"
        # "--dataset palm4748a_50erd"
        # "--dataset palm4748a_50erd --tag a"
        # "--dataset palm4748a_50erd --tag b"
        # "--dataset palm4748a_50erd --tag c"
        # "--dataset palm4748a_50optt --tag sgdb"
        # "--dataset palm4748a_50optt --tag sgdc"
        # "--dataset palm4748a_50optt --tag sgdd"
        "--optimizer=adam --lr=2.5e-4 --tag adam"
        # "--optimizer=adam --lr=1e-5"
        #"--optimizer=sgd --lr=2.5e-4 --momentum=0.9 --lr-step=100000"
        #"--optimizer=sgd --lr=1e-5 --momentum=0.9 --lr-step=100000"
        # "--optimizer=sgd --lr=1e-4 --momentum=0.9 --lr-step=100000"
        # "--optimizer=sgd --lr=1e-4 --momentum=0.9 --lr-step=50000"
        # "--optimizer=sgd --lr=2.5e-4 --momentum=0.9 --lr-step=50000"
        # "--optimizer=sgd --lr=1e-3 --momentum=0.9 --lr-step=100000" leads to nan
        # "--tag optimizersg"
        # "--tag optimizers"
        # '--lr=1e-4 --momentum=0.9'
        # '--lr=5e-5 --momentum=0.9'
        # '--lr=5e-5 --momentum=0.9'
#'--active-samples=10 --rand-option=ar'
#'--active-samples=30 --rand-option=a'
#'--active-samples=30 --rand-option=c'
# 
        #  '--active-samples=30 --rand-option=b'
        #  '--active-samples=30 --rand-option=br'
        #  '--active-samples=50 --rand-option=b'
        #  '--active-samples=50 --rand-option=br'
        #  '--active-samples=50 --rand-option=c'
        #  '--active-samples=50 --rand-option=cr'
        )
others2=('--is-use-location --fusion-type=soft')
#EXTRA_ARGS='--not-save-arrays --optimizer=adam --lr=2.5e-4 --lambda-reg=0.5 --sq-kernel=2' # --is-overwrite'
## for palm4748a
# EXTRA_ARGS='--tag=sept29 --train-patches=100000 --is-total-patches-datasets --eval-every 5 --epochs=200 --optimizer=sgd --lr=1e-4 --momentum=0.9 --lr-step=100000 --patch-size=16 --batch-size=128 --not-save-arrays --lambda-reg=0.5 --sq-kernel=2' # --is-save-data-only' #--is-val' # --is-overwrite'

## for palm4
#EXTRA_ARGS='--train-patches=1000000 --is-total-patches-datasets --val-patches=200 --eval-every=5 --epochs=100 --optimizer=sgd --lr=1e-4 --momentum=0.9 --lr-step=500000 --patch-size=16 --batch-size=128 --not-save-arrays --lambda-reg=0.5 --sq-kernel=2' #--is-val' # --is-overwrite'
EXTRA_ARGS='--train-patches=1000000 --is-total-patches-datasets --val-patches=200 --eval-every=5 --epochs=100 --patch-size=16 --batch-size=128 --not-save-arrays --lambda-reg=0.5 --sq-kernel=2' #--is-val' # --is-overwrite'

# --is-save-data-only 

#BSUB -W 24:00
#BSUB -o /cluster/scratch/andresro/sparse/output/train_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/sparse/output/train_Le.%J.%I.txt
#BSUB -R "rusage[mem=64000,ngpus_excl_p=1]"
#BSUB -R "select[gpu_mtotal0>=10240]"
#BSUB -n 1
##BSUB -N
#BSUB -J palmadam[1]
##BSUB -w 'numended(11270372,*)'
##BSUB -w 'numdone(10279448,*)'
##BSUB -u andresro@ethz.ch
 
#### BEGIN #####

index=$((LSB_JOBINDEX-1))

set +x
i=0
  for mod in "${models[@]}"; do
   for d in "${datasets[@]}"; do
    for oth in "${others[@]}"; do
     for oth1 in "${others1[@]}"; do
      for oth2 in "${others2[@]}"; do
	  if [ "$index" -eq "$i" ]; then
                set -x
                MODEL=$mod
                DATASET=$d
                OTHER=$oth' '$oth1' '$oth2
                set +x
        fi
        ((i+=1))
done;done;done;done;done;
echo $i combinations

set -x

#PATCH=16 # ${patches[index]}
# BATCH=128
if [ "$index" -gt "$i" ]; then
exit 1;fi



#version=${OPTIM}${LRATE}

# #WARM=$SCRATCH/sparse/training/snapshots/palm/$DATASET/$MODEL/PATCH16_16_SCALE2_Lr0.5_Lw0.0000_sq2adam2.5e-4
# WARM=$SCRATCH/sparse/training/snapshots/palm/palmpeninsulanew/simpleA9/PATCH16_16_SCALE2_Lr0.5_Lw0.0000_sq2adam2.5e-4_clean

set +x
#module load python_gpu/2.7.14 gdal/2.2.2 cudnn/7.3
#module load python_gpu/3.7.1  gdal/2.4.4  cudnn/7.6.4
module load python_gpu/3.7.1  gdal/3.1.2  cudnn/7.6.4


set -x


cd $HOME/code/sparsem
python3 -u main.py \
    --dataset=$DATASET \
    --model=$MODEL \
    --logsteps=500 \
    --numpy-seed=1234 \
    --patch-size-eval=16 --batch-size-eval=128 ${EXTRA_ARGS} $OTHER
    # --is-overwrite \
    # --scale=$SCALE \
    # --lambda-weights=$LAMW \
    # --lambda-sr=$LAMSR 
    # --warm-start-from=$WARM \

#--unlabeled_data=$WORK/barry_palm/data/2A/palmcountries_2017/S2A_MSIL2A_20171115T023951_N0206_R089_T49NFC_20171115T075621.SAFE  --roi_lon_lat=112.5,2.2,112.6,2.4 --semi-supervised=semiRev
#    --unlabeled_data=$WORK/barry_palm/data/2A/socb_2018/S2A_MSIL2A_20180428T104021_N0206_R008_T29NQF_20180428T155845.SAFE --roi_lon_lat_unlab=-7.2360689999999996,4.5388549999999999,-7.0136520000000004,4.7923210000000003 --semi-supervised=semiRev

# --is-degraded-hr 
#    --is-hr-label
#    --is-same-volume
#
#
#### END #####
