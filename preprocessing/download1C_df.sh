#!/usr/bin/env bash
now="$(date +'%d-%m-%Y_%H-%M')"


if [ -z "$WORK" ]
then
      echo "\$WORK is setting to pf share"
      WORK=$PF
fi

CONFIG=$1
EXTRAARGS=${@:2}
case ${CONFIG} in
  phillipines_2017)
    SAVE_FOLDER=phillipines_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="phillipines_2017/Phillipines_all_1840.pkl"
    ;;
  phillipines_small_2017)
    SAVE_FOLDER=phillipines_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="phillipines_2017/Palawan_Cebu_Davao_del_Norte_Davao_del_Sur_Davao_Oriental_550.pkl"
    ;;
  indonesia_2017)
    SAVE_FOLDER=palmcountries_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="palmcountries_2017/Indonesia_all_8410.pkl"
    ;;
  indonesia_medium_2017)
    SAVE_FOLDER=palmcountries_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="palmcountries_2017/Kalimantan_Barat_Riau_Sulawesi_Barat_Sulawesi_Tengah_Sulawesi_Utara_Gorontalo_1651.pkl"
    ;;
  palawan_2017)
    SAVE_FOLDER=phillipines_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="phillipines_2017/Palawan_320.pkl"
    ;;
  indonesia_small_2017)
    SAVE_FOLDER=palmcountries_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="palmcountries_2017/Kalimantan_Barat_Riau_791.pkl"
    ;;
  malaysia_2017)
    SAVE_FOLDER=palmcountries_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="palmcountries_2017/Malaysia_all_1150.pkl"
    ;;
  malaysia_small_2017)
    SAVE_FOLDER=palmcountries_2017
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/1C/
    df_name="palmcountries_2017/Sarawak_440.pkl"
    ;;
  malaysia_2019)
    SAVE_FOLDER=asia_2019
    PATH_OUTPUT=$WORK/barry_palm/output/
    path_data=$WORK/barry_palm/data/2A/
    df_name="asia_2019/Malaysia_all_1151.pkl"
    ;;
   all)
    bash download1C_df.sh phillipines_2017 ${@:2}
    bash download1C_df.sh indonesia_2017 ${@:2}
    bash download1C_df.sh malaysia_2017 ${@:2}
    exit
    ;;
  *)
    echo "Option not defined"
    exit
    ;;
esac


module load python_gpu/3.7.1

exec &> >(tee -a "${PATH_OUTPUT}/download_1C_$now.txt")


echo $CONFIG

read -p 'Username: ' user
read -sp "Password: " password
echo ""
mkdir -p $path_data/$SAVE_FOLDER

df_path=$path_data/dataframes_download/$df_name


python3 -u download1C_df.py --username=$user --password=$password --df-path $df_path --save-dir $path_data/$SAVE_FOLDER/PRODUCT/ $EXTRAARGS
