#!/bin/bash
now="$(date +'%d-%m-%Y_%H-%M')"

# PATH_PALM=$1        #/scratch/andresro/palm
# PATH_sen2cor=$2     #/scratch/andresro/apps/Sen2Cor-2.4.0-Linux64/bin
# file=$3             #S2A_MSIL1C_20161225T033142_N0204_R018_T47MRV_20161225T034118.zip
LOC=asia_2019
overwrite=0        # 0: no overwrite, 1 for overwriting if 2A folder exists, 2 for overwriting all


PATH_PALM=${WORK}/barry_palm

PATH_RAW=${PATH_PALM}/data/2A/$LOC/PRODUCT
PATH_DATA=${PATH_PALM}/data/2A/$LOC

type='zip'


cd $PATH_RAW

# CHECK IF 2A FILES ALREADY EXIST, COLLECT LIST OF 1C_FILES TO PROCESS
i=0
for file in *.${type}; do
    FILENAME_1C="${file/.$type/}"
    #echo file $file
    if grep -Fxq "$FILENAME_1C" ${PATH_RAW}/correct_zip.txt; then
        if [ ! -d "${PATH_DATA}/${FILENAME_1C}.SAFE" ] || [ "$overwrite" -gt 1 ] ; then
                echo unzipping $FILENAME_1C.zip
                START=$(date +%s.%N)
                unzip -q -o $PATH_RAW/${FILENAME_1C}.zip -d $PATH_DATA/
                END=$(date +%s.%N)
                echo Time unzip $(echo "$END - $START" | bc)
        # else
            # echo "already unzipped" $FILENAME_1C.SAFE
        fi
    else
        echo $FILENAME_1C "not found in correct_zip"
    fi
done;
#    mv $PATH_RAW/${FILENAME_2A}.SAFE/ $PATH_DATA/
#    rm -r $PATH_RAW/${FILENAME_1C}.SAFE



