#!/bin/bash

# Run ls command and capture the output
ROOT="/usace_share/SewerML/"

TRAINS=(train00 train01 train02 train03 train04 train05 train06 train07 train08 train09 train10 train11 train12 train13)
VALID=(valid00 valid01)
TEST=(ttest00 ttest01)


for FOLDER in train00 train01 train02 train03 train04 train05 train06 train07 train08 train09 train10 train11 train12 train13
do
  output=$(ls $ROOT/$FOLDER | wc -l)
  result=$((output / 100))
  echo "[+] $FOLDER = $result"
  time python main_sample_dataset.py --folder $FOLDER --num-samples $result --outdir dataset/train
done
