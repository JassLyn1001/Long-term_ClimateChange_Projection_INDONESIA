#!/usr/bin/bash

# --- This script contains some codes to execute python script within a Bash for loop ---

models=("ACCESS-CM2" "ACCESS-ESM1-5" "BCC-CSM2-MR" "CESM2" "CESM2-WACCM" "CMCC-CM2-SR5" "CMCC-ESM2" " CNRM-CM6-1" "CNRM-ESM2-1" "CanESM5" "EC-Earth3" "EC-Earth3-Veg-LR" "FGOALS-g3" "GFDL-CM4" "GFDL-ESM4" "GISS-E2-1-G" "HadGEM3-GC31-LL" "HadGEM3-GC31-MM" "IITM-ESM" "INM-CM4-8" "INM-CM5-0" "IPSL-CM6A-LR " "KACE-1-0-G" "KIOST-ESM" "MIROC-ES2L" "MIROC6" "MPI-ESM1-2-HR" "MPI-ESM1-2-LR" "MRI-ESM2-0" "NESM3" "NorESM2-LM" "NorESM2-MM" "TaiESM1" "UKESM1-0-LL")

for model in ${models[@]}
do
    echo "Processing python file for  ${model}"
    python bc_tas_ssp245_LinearRegTF.py ${model}
done
