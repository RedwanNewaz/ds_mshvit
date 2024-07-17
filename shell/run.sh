#!/bin/bash
#dataset
RGB_ROOT='/home/aredwann/MSHViT/dataLink/rgb_train_val'
OPTI_ROOT='/home/aredwann/MSHViT/dataLink/train_val'

# pretrained weights
baseWeight='/home/aredwann/MSHViT/weights/resnet50_mshvit.ckpt'
optiWeight='/home/aredwann/MSHViT/results/opticalflow_full/resent50_msvit_opticalflow.ckpt'

# specific to this project
PROJECT_ROOT="/home/aredwann/PyDev/SewerML-tracker"
python='/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python'
ann_root="$PROJECT_ROOT/annotations_sewerml"
output="$PROJECT_ROOT/results/fuseNets"


CUDA_VISIBLE_DEVICES=3 $python $PROJECT_ROOT/main.py --ann_root $ann_root --rgb_data_root $RGB_ROOT  --opti_data_root $OPTI_ROOT --results_output $output \
    --baseline_model_path $baseWeight  --optical_model_path $optiWeight --batch_size 128
