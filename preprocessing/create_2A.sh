#!/bin/bash
now="$(date +'%d-%m-%Y_%H-%M')"

PATH_PALM=$1        #/scratch/andresro/palm
PATH_sen2cor=$2     #/scratch/andresro/apps/Sen2Cor-2.4.0-Linux64/bin
file=$3             #S2A_MSIL1C_20161225T033142_N0204_R018_T47MRV_20161225T034118.zip
LOC=$4              #indonesia
overwrite=$5        # 0: no overwrite, 1 for overwriting if 2A folder exists, 2 for overwriting all


PATH_RAW=${PATH_PALM}/data/1C/$LOC/PRODUCT
PATH_DATA=${PATH_PALM}/data/2A/$LOC

echo $now
echo reading data from $PATH_RAW

set -x
set -e
FILENAME_1C=$file
if [[ "${FILENAME_1C}" = *".zip" ]] ; then
  FILENAME_1C="${FILENAME_1C/.zip/}"
fi

if [[ "${FILENAME_1C}" = *".SAFE" ]] ; then
  FILENAME_1C="${FILENAME_1C/.SAFE/}"
fi


FILENAME_2A="${FILENAME_1C/1C_/2A_}"

if [[ "${FILENAME_2A}" = *"_OPER_"* ]] ; then  # replace _OPER_ if exists in string
  FILENAME_2A="${FILENAME_2A/_OPER_/_USER_}"
fi

if [ ! -f "$PATH_RAW/$FILENAME_1C.zip" ] && [ ! -d "$PATH_RAW/$FILENAME_1C.SAFE" ]; then
    echo "${PATH_RAW}/${FILENAME_1C} .zip or .SAFE not found!"
    exit 0
fi

if [ ! -d "$PATH_DATA/${FILENAME_2A}.SAFE" ] || [ "$overwrite" -gt 0 ] ; then

    # rm old 2A if exists
    rm -rf $PATH_DATA/${FILENAME_2A}.SAFE

    ## UNZIP if necessary or overwrite
    cd $PATH_RAW
    if [ -f "$PATH_RAW/$FILENAME_1C.zip" ]; then
        if [ ! -d "${PATH_RAW}/${FILENAME_1C}.SAFE" ] || [ "$overwrite" -gt 1 ] ; then
            echo unzipping $FILENAME_1C.zip
            START=$(date +%s.%N)
            unzip -o $PATH_RAW/${FILENAME_1C}.zip
            END=$(date +%s.%N)
            echo Time unzip $(echo "$END - $START" | bc)
        fi
    else
    	echo using alread unzipped $FILENAME_1C.SAFE
    fi

    ## CREATE 2A PRODUCT
    cd $PATH_sen2cor

    START=$(date +%s.%N)
        ./L2A_Process $PATH_RAW/$FILENAME_1C.SAFE --output_dir $PATH_DATA/
    END=$(date +%s.%N)
    echo Time L2A_Proces $(echo "$END - $START" | bc)

    echo FILENAME_2A: $FILENAME_2A
#    mv $PATH_RAW/${FILENAME_2A}.SAFE/ $PATH_DATA/
#    rm -r $PATH_RAW/${FILENAME_1C}.SAFE

else
    echo $PATH_DATA/${FILENAME_2A}.SAFE already exists
    echo $PATH_DATA/${FILENAME_2A}.SAFE $now >> $PATH_DATA/existing2A.log
fi



