export CUDA_HOME=~/anaconda3/envs/SewerML-tracker3
version='32'
ROOT="/home/aredwann/PyDev/SewerML-tracker"
ann_root="$ROOT/annotations_sewerml/Johny_091023"
data_root="$ROOT/annotations_sewerml/Johny_old/data"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/baseline_inference.py"
# PY_FILE="/home/aredwann/MSHViT/inference.py"
METRIC_PY_FILE="$ROOT/metrics/calculate_results.py"
dev_model_weight="$ROOT/results/joint_vit/weights/SewerML_RGB_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn_RGB_version_$version"
# baseline_model_weight="/home/aredwann/MSHViT/weights/resnet50_hvit_sinkhorn"


computeResult()
{
    echo "[+] computing results for $2 in CUDA $3"
    model_path=$1
    results_output="$ROOT/results/rgb_vit/$2"
    mkdir -p $results_output

    CUDA_VISIBLE_DEVICES=$3 $PYTHON $PY_FILE --model_path $model_path \
                    --model_version "benchmark" \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_val 

    $PYTHON $METRIC_PY_FILE --score_path $results_output --split Val --gt_path $ann_root --output_path $results_output
}


# weights=(
#     "botnet"
#     "botnet_mshvit"
#     "coatnet0"
#     "coatnet0_mshvit"
#     "coatnet1"
#     "coatnet1_mshvit"
#     "resnet101"
#     "resnet101_mshvit"
#     "resnet18"
#     "resnet18_mshvit"
#     "resnet34"
#     "resnet34_mshvit"
#     "resnet50"
#     "resnet50_hvit_patch"
#     "resnet50_mshvit"
#     "tresnetl"
#     "tresnetl_mshvit"
#     "tresnetm"
#     "tresnetm_mshvit"
# )

weights=(
    "tresnetm_mshvit"
)

TOTAL=18

START=$1
model=${weights[$START]}
baseline_model_weight="/home/aredwann/MSHViT/weights/$model.ckpt"
computeResult $baseline_model_weight $model $START

