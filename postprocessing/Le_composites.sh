#!/usr/bin/env bash

set -x
set -e



DATADIR=$WORK/barry_palm/data/2A/asia_2020after/PRODUCT/
SAVEDIR=$WORK/barry_palm/data/2A/asia_2020after/composite/
# 53
tiles=(T51PWM T51PVR T51QTA T51PWR T51QVU T50PRV T51PUT T51PVP T51PUM T51PWQ T51PTS T51PWP T51PVQ T50PRC T51PXM T51PTP T51PUQ T51PXL T51PYP T51PWN T51QTU T51PTQ T51QUU T51PTR T50QRF T51PUP T51PUS T51PXP T51QVA T51PXR T51QVV T50QRD T51PVL T51QUA T51PTT T51PUN T51QUV T50PRB T51PXQ T51PYQ T51PVN T50QRE T50PRA T51PUR T51PYM T51PYN T50QQD T51PXN T51PWL T51PVM T51QTV T51PVS T50PQC)


#BSUB -W 4:00
#BSUB -o /cluster/scratch/andresro/sparse/output/agg_pred_Le.%J.%I.txt
#BSUB -e /cluster/scratch/andresro/sparse/output/agg_pred_Le.%J.%I.txt
#BSUB -R "rusage[mem=64000]"
#BSUB -n 2
##BSUB -N
#BSUB -J aggregate[1]
##BSUB -w 'numended(11706259,*)'
#BSUB -u andresro@ethz.ch

#### BEGIN #####

index=$((LSB_JOBINDEX-1))

set +x


TILE=${tiles[index]}

set -x


# DATADIR=$modeldir/$TILE

set +x

# module load python_cpu/3.7.1  gdal/2.4.4
module load python_cpu/3.7.1  gdal/3.1.2


set -x

python3 -u composite_per_tile.py $DATADIR $TILE --save-dir $SAVEDIR --is-overwrite --compression=12 --function mean

#
#### END #####
