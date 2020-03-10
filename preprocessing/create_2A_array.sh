#!/bin/bash

# SET PARAMETERS

project_name=barry_palm
loc=nigeria_2017
#path_code=/cluster/home/${USER}/code/sentinel
path_code=$HOME/code/sen2patches
overwrite=2 # 0: no overwrite, 1 for overwriting if 2A folder exists, 2 for overwriting all

path_output=${HOME}/${project_name}/output/
path_sen2cor=${HOME}/apps/Sen2Cor-2.4.0-Linux64/bin

path_data_parent=${WORK}/${project_name}
path_data=${path_data_parent}/data


#BSUB -W 24:00
#BSUB -o $path_output/Eu_create2A.%J.%I.txt
#BSUB -e $path_output/Eu_create2A.%J.%I.txt
#BSUB -R "rusage[mem=10000,scratch=10000]"
#BSUB -n 1
#BSUB -N
#BSUB -J "2A[1-154]"
##BSUB -w 'numended(53637271,*)'
#### BEGIN #####

# HOW TO USE THIS SCRIPT:
# 1. SET THE NUMBER OF JOBS RUNNING IN PARALLEL IN ##BSUB -J "2A[1-${NUMER_OF_JOBS}]"
#       THE NUMBER OF JOBS EQUAL THE NUMBER OF FILES TO PROCESS
# 2. SET PARAMETERS

# LOAD THE MODULES
module load python_cpu/2.7.14 # was python/2.7.6

cd ${path_data}/1C/$loc/PRODUCT

# CHECK IF 2A FILES ALREADY EXIST, COLLECT LIST OF 1C_FILES TO PROCESS
i=0
for file in *.zip; do
    FILENAME_1C="${file/.zip/}"
    #echo file $file
    #echo FILENAME_1C $FILENAME_1C
    FILENAME_2A="${FILENAME_1C/1C_/2A_}"
    if [[ "${FILENAME_2A}" = *"_OPER_"* ]] ; then  # replace _OPER_ if exists in string
  	FILENAME_2A="${FILENAME_2A/_OPER_/_USER_}"
    fi

    if [ ! -d "${path_data}/2A/$loc/${FILENAME_2A}.SAFE" ] || [ "$overwrite" -gt 0 ] ; then
        result=$result,$FILENAME_1C
        i=$((i+1))
    fi
done

echo length $i # change the size of the array after knowing the length

parameters=$result

# SAVE THE FILENAMES TO PROCESS IN AN ARRAY
IFS=', ' read -r -a array_param <<< "$parameters"

set -x
# FOR EACH JOB (LSB_JOBINDEX) GET THE FILENAME TO PROCESS FROM THE ARRAY
#index=$((LSB_JOBINDEX-1))

index=$((LSB_JOBINDEX))
echo processing file
file=${array_param[index]}

# PROCESS THE FILE IN EACH JOB
cd $path_code

./create_2A_euler.sh ${path_data_parent} ${path_sen2cor} $file $loc $overwrite
#
#
#
#### END #####
