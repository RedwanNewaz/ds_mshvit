#!/usr/bin/env bash
export CUDA_HOME=~/anaconda3/envs/SewerML-tracker3
MAJOR_VERSION=11
# Check if an argument was passed to the script
if [ "$#" -eq 0 ]; then
  echo "[!] No argument was passed. Specify a valid integer for log MINOR_VERSION."
  exit 1
else
  # Get the argument
  argument=$1

  # Check if the argument is a valid integer
  if [[ ! "$argument" =~ ^[0-9]+$ ]]; then
    echo "[-] Minor version argument must be a valid integer."
    exit 1
  fi

  if [ "$#" -eq 2 ]; then
    if [[ ! "$2" =~ ^[0-9]+$ ]]; then
      echo "[-] Major version argument must be a valid integer."
      exit 1
    else
      MAJOR_VERSION=$2
    fi
  fi 

  # Print the feedback
  echo "[+] The training VERSION argument is $MAJOR_VERSION.$argument."
fi


# related to training param
NUM_GPUS=1


MINOR_VERSION=$argument
SEED=147049227

# related to filesystem directories
RGB_DATA_DIR_TRAIN="/usace_share/Share_Redwan/375_374_D/rgb_img640x480"
OPTI_DATA_DIR_TRAIN="/usace_share/Share_Redwan/375_374_D/opti_img_640x480"
ANNOTATION_DIR_TRAIN="/usace_share/Share_Redwan/375_374_D"

RGB_DATA_DIR_VAL="/usace_share/Share_Redwan/377_376_D/rgb_img640x480"
OPTI_DATA_DIR_VAL="/usace_share/Share_Redwan/377_376_D/opti_img_640x480"
ANNOTATION_DIR_VAL="/usace_share/Share_Redwan/377_376_D"



OUTPUT_DIR="/home/aredwann/PyDev/SewerML-tracker/results/375_374_D"
ROOT_DIR="/home/aredwann/PyDev/SewerML-tracker"
# select which GPU you want to use 
DEVICE_NUM=3

#remaining backbones: coatnet_1, resnet_18, resnet_34
# Resources/Dataset/SewerML/Model/CoAtNet-1
BATCH_SIZE=256 # default 256
TOKEN_DIM=512 # default 512
BACKBONE_NET="resnet101"
OUTPUT_MODEL="resnet101"
#python 
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"


extract_model()
{
  version="$MAJOR_VERSION$MINOR_VERSION"
  modelPath="${ROOT_DIR}/results/joint_vit/SewerMLJoint_${BACKBONE_NET}_MultiScaleViTHead_FNetBlock_Sinkhorn/version_$version"
  outputDir="${ROOT_DIR}/Resources/Dataset/SewerML/Model/${OUTPUT_MODEL}/DS-MSHViT/Weight"
  mkdir -p $outputDir
  modelName="DS-MSHViT-${OUTPUT_MODEL}"

  printf "[+] model = %s \n[+] outdir = %s \n[+] rename = %s \n" $modelPath $outputDir $modelName 


  EXTRACT_PY_FILE="/home/aredwann/PyDev/SewerML-tracker/metrics/extract_weights.py"
  
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM  $PYTHON $EXTRACT_PY_FILE  --model_path $modelPath \
                    --output_dir $outputDir \
                    --model_name $modelName


  SEPERATE_PY_FILE="/home/aredwann/PyDev/SewerML-tracker/metrics/seperate_rgb_weights.py"
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM  $PYTHON $SEPERATE_PY_FILE  --model_path $modelPath \
                    --output_dir $outputDir \
                    --model_name "$modelName-RGB"
}


train()
{
  # related to network

  HEAD_NET="MultiScaleViTHead"

  # related to weights and biases website
  WB_PROJECT="sewer_ml_joint_vit_v$MAJOR_VERSION.$MINOR_VERSION"
  WB_GROUP="sewer_ml_joint_vit_v$MAJOR_VERSION.$MINOR_VERSION"

  # use python from the conda environment
  PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
  TRAIN_PY="/home/aredwann/PyDev/SewerML-tracker/adaptive_trainer_v4.py"

  CUDA_VISIBLE_DEVICES=$DEVICE_NUM $PYTHON $TRAIN_PY \
          --precision 16 \
          --batch_size $BATCH_SIZE \
          --max_epochs 40 \
          --gpus $NUM_GPUS \
          --img_size 224 \
          --ann_root_train $ANNOTATION_DIR_TRAIN \
          --rgb_data_root_train $RGB_DATA_DIR_TRAIN \
          --opti_data_root_train $OPTI_DATA_DIR_TRAIN \
          --ann_root_val $ANNOTATION_DIR_VAL \
          --rgb_data_root_val $RGB_DATA_DIR_VAL \
          --opti_data_root_val $OPTI_DATA_DIR_VAL \
          --deterministic \
          --backbone_model $BACKBONE_NET \
          --head_model $HEAD_NET \
          --token_dim $TOKEN_DIM \
          --transformer_depth 2 \
          --block_drop 0. \
          --tokenizer_layer Sinkhorn \
          --block_type FNetBlock \
          --use_pos_embed \
          --num_heads 8 \
          --qkv_bias \
          --mlp_ratio 4. \
          --proj_drop 0. \
          --attn_drop 0. \
          --pos_embed_drop 0. \
          --cross_num_heads 8 \
          --cross_qkv_bias \
          --cross_proj_drop 0. \
          --cross_attn_drop 0. \
          --cross_block_drop 0. \
          --num_cluster 64 \
          --sinkhorn_eps 0.25 \
          --sinkhorn_iters 5 \
          --l2_normalize \
          --backbone_feature_maps layer3 layer4 \
          --shared_tower \
          --multiscale_method CrossScale-Token \
          --sigmoid_loss \
          --seed $SEED \
          --wandb_project $WB_PROJECT \
          --wandb_group $WB_GROUP \
          --progress_bar_refresh_rate 1000 \
          --flush_logs_every_n_steps 1000 \
          --log_version "$MAJOR_VERSION$MINOR_VERSION" \
          --log_every_n_steps 1 \
          --log_save_dir $OUTPUT_DIR \
          --save_last

          # --model_ema \
          # --monitor_metric \
}


train
# extract_model



