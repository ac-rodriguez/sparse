#!/bin/bash

set -x
set -e

#tiles=(R018_T47NQA R046_T50PNQ R132_T49MCV R132_T49MDV)
#tiles=(R046_T50PNQ R046_T50PPR R046_T50PNR)
#tiles=(R046_T50PPR R046_T50PNR R132_T49NCA R132_T49NDA)
#tiles=(T49NCB T49NDB T49NEB)
#tiles=(T49MDV T49MCV T47NQA T50PNQ)
#tiles=(T50PNQ T50PNR T47NQA T49MCV T49MDV T49NCA T49NDA) 
#tiles=(T49NCB T49NDB T49NEB)
#tiles=(T49MCV)
#tiles=(T49NED) #T49NEC T49NFC T49NFD T49NGD)

#tiles=(T49NED T49NEC T49NFC T49NFD T49NGD T49MCV)
#tiles=(T49NFC T49NFD T49NGD)
# Tiles sarawak
#tiles=(T49NCB T49NCC T49NDA T49NDB T49NDC T49NEB T49NEC T49NED T49NFB T49NFC T49NFD T49NGB T49NGC T49NGD T49NGE T49NHB T49NHC T49NHD T49NHE T49NHF T50NKG T50NKH T50NKJ T50NKK T50NKL T50NLH T50NLJ T50NLK T50NLL)
# tiles peninsula
#tiles=(T47NNG T47NNH T47NPE T47NPF T47NPG T47NPH T47NQC T47NQD T47NQE T47NQF T47NQG T47NQH T47NRC T47NRD T47NRE T47NRF T47NRG T47NRH T48NTG T48NTH T48NTJ T48NTK T48NTL T48NTM T48NUG T48NUH T48NUJ T48NUK T48NUL T48NVG T48NVH T48NVJ)
#tiles=(T47NRE T48NTK T48NTJ)
#tiles=(T47NRE T47NQD T48NUH)

#tiles north
#tiles=(T47NQD T48NTK T48NTJ T48NUH)
#tiles=(T47NQD T48NUH T48NTL)
# tiles sabah
#tiles=(T50NLK T50NLL T50NLM T50NMK T50NML T50NMM T50NMN T50NMP T50NNK T50NNL T50NNM T50NNN T50NNP T50NPK T50NPL T50NPM T50NPN T50NQK T50NQL T50NQM)

# RIAU
#tiles=(T47MPV T47MQU T47MQV T47MRU T47MRV T47NPA T47NPB T47NPC T47NQA T47NQB T47NQC T47NRA T47NRB T47NRC T48MTD T48MTE T48MUE T48NTF T48NTG T48NUF)
#modeldir=$WORK/sparse/inference/palmriau_simpleA9all

#modeldir=$WORK/sparse/inference/palmsabah_simpleA9all

#modeldir=$WORK/sparse/inference/palmpeninsulanew_simpleA9all


tiles=(T49MCV)
modeldir=$WORK/sparse/inference/palmsarawak3_simpleA20westkalim

#modeldir=$SCRATCH/sparse/inference/palmcoco_kalimA_simpleA5 #palmtiles_simpleA #cococomplete_simpleA
#modeldir=$SCRATCH/sparse/inference/palmsarawak_simpleA9fix #palmtiles_simpleA #cococomplete_simpleA
#modeldir=$SCRATCH/sparse/inference/palmcocotiles2_coco_palm_kalim_simpleA
#modeldir=$SCRATCH/sparse/inference/palmsarawak_simpleA20_allsarawak
#modeldir=$SCRATCH/sparse/inference/palmsabah_simpleA9
#tiles=(T50NNL T50NNM)
#modeldir=$SCRATCH/sparse/inference/palmpeninsula2_simpleA

#modeldir=$WORK/sparse/inference/palmsarawak3_simpleA20clean_all
#modeldir=$WORK/sparse/inference/palmpeninsula2_simpleAall
#tiles=(T47NRE T48NTJ)

#BSUB -W 4:00
#BSUB -o /cluster/home/andresro/code/output_sparse/pred_Le.%J.%I.txt
#BSUB -e /cluster/home/andresro/code/output_sparse/pred_Le.%J.%I.txt
#BSUB -R "rusage[mem=64000]"
#BSUB -n 2
#BSUB -N
#BSUB -J aggregate[1]
#BSUB -w 'numended(2959352,*)'
#BSUB -u andresro@ethz.ch
 
#### BEGIN #####

index=$((LSB_JOBINDEX-1))

set +x


TILE=${tiles[index]}

set -x


DATADIR=$modeldir/$TILE

set +x

module load python_cpu/3.7.1  gdal/2.2.2 

set -x


cd $HOME/code/sparsem
python3 -u aggregate_per_tile.py $DATADIR --is-avgprobs --is-overwrite --is-clip-to-countries --is-reg-only --is-remove-water

#
#### END #####
