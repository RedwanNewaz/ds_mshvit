#!/usr/bin/env bash

MAJOR_VERSION=5
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
BATCH_SIZE=256
TOKEN_DIM=512
MINOR_VERSION=$argument
SEED=147049227

# related to filesystem directories
RGB_DATA_DIR="/home/aredwann/MSHViT/dataLink/rgb_train_val"
OPTI_DATA_DIR="/home/aredwann/MSHViT/dataLink/train_val"
# ANNOTATION_DIR="/home/aredwann/PyDev/SewerML-tracker/annotations_sewerml"
ANNOTATION_DIR="/home/aredwann/MSHViT/annotations_sewerml"
OUTPUT_DIR="/home/aredwann/PyDev/SewerML-tracker/results/seq_vit"
mkdir -p $OUTPUT_DIR

# related to network
BACKBONE_NET="resnet50"
HEAD_NET="MultiScaleViTHead"

# # pretrained weights
PRETRAINED_BACKBONE="/home/aredwann/MSHViT/weights/resnet50_hvit_sinkhorn.ckpt"


# related to weights and biases website
WB_PROJECT="sewer_ml_joint_vit_v$MAJOR_VERSION.$MINOR_VERSION"
WB_GROUP="sewer_ml_joint_vit_v$MAJOR_VERSION.$MINOR_VERSION"

# use python from the conda environment
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
TRAIN_PY="/home/aredwann/PyDev/SewerML-tracker/sequence_trainer_v2.py"

CUDA_VISIBLE_DEVICES=3 $PYTHON $TRAIN_PY \
        --precision 16 \
        --batch_size $BATCH_SIZE \
        --max_epochs 40 \
        --gpus $NUM_GPUS \
        --img_size 224 \
        --ann_root $ANNOTATION_DIR \
        --data_root $RGB_DATA_DIR \
        --deterministic \
        --backbone_model $BACKBONE_NET \
        --head_model $HEAD_NET \
        --pretrained_backbone $PRETRAINED_BACKBONE \
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
        --backbone_feature_maps layer2 layer3 \
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