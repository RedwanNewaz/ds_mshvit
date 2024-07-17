version='32'
threshold='00'
alpha=0.00
ROOT="/home/aredwann/PyDev/SewerML-tracker"
model_path="$ROOT/results/joint_vit/weights/SewerMLJoint_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn_version_$version.ckpt"
model_version="$version$threshold"
results_output="$ROOT/results/joint_vit/prediction/$model_version"
ann_root="$ROOT/annotations_sewerml"
rgb_data_root="/home/aredwann/MSHViT/dataLink/rgb_val"
opti_data_root="/home/aredwann/MSHViT/dataLink/val"

mkdir -p $results_output

PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/inference.py"


# CUDA_VISIBLE_DEVICES=2 $PYTHON $PY_FILE --model_path $model_path \
#                 --model_version $model_version \
#                 --results_output $results_output \
#                 --rgb_data_root $rgb_data_root\
#                 --opti_data_root $opti_data_root \
#                 --ann_root $ann_root \
#                 --alpha $alpha \
#                 --dataset SewerMLJoint \
#                 --eval_val

PY_FILE="$ROOT/metrics/calculate_results.py"

$PYTHON $PY_FILE --score_path $results_output --split Val --gt_path $ann_root --output_path "$ROOT/results/joint_vit/$model_version"