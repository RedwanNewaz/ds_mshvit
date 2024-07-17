export CUDA_HOME=~/anaconda3/envs/SewerML-tracker3
ROOT="/home/aredwann/PyDev/SewerML-tracker"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/baseline_inference.py"
DATASET="SewerML"
DEVICE=3

computeResult_DS_MSHViT()
{
    model=$1
    ann_root="$ROOT/Resources/Dataset/${DATASET}/Annotation/Validation"
    data_root="$ROOT/Resources/Dataset/${DATASET}/Data/Validation/val_data"
    model_path="${ROOT}/Resources/Dataset/SewerML/Model/${model}/DS-MSHViT/Weight/DS-MSHViT-${model}-RGB.ckpt"
    results_output="${ROOT}/Resources/Dataset/${DATASET}/Model/${model}/DS-MSHViT/Prediction"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON $PY_FILE --model_path $model_path \
                    --model_version DS-MSHViT-${model}-RGB_Val \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_val 
    echo "[+] computing results for $1 in CUDA $DEVICE is finished !!!"
}

computeResult_MSHViT()
{
    model=$1
    weight=$2
    ann_root="$ROOT/Resources/Dataset/${DATASET}/Annotation/Validation"
    data_root="$ROOT/Resources/Dataset/${DATASET}/Data/Validation/val_data"
    model_path="${ROOT}/Resources/Dataset/SewerML/Model/${model}/MSHViT/Weight/${weight}_mshvit.ckpt"
    results_output="${ROOT}/Resources/Dataset/${DATASET}/Model/${model}/MSHViT/Prediction"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON $PY_FILE --model_path $model_path \
                    --model_version MSHViT-${model}_${DATASET}_Val \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_val 
    echo "[+] computing results for $1 in CUDA $DEVICE is finished !!!"
}

# computeResult_DS_MSHViT ResNet-101

# computeResult_MSHViT ResNet-101 resnet101

mshvit_models=(
    "botnet"
    "coatnet0"
    "resnet18"
    "resnet34"
    "tresnetl"
    "tresnetm"
)

ds_mshvit_models=(
    "BotNet-50-S1"
    "CoAtNet-0"
    "ResNet-18"
    "ResNet-34"
    "TResNet-L"
    "TResNet-M"
)

for i in {0,1,2,3,4,5}
do 
    mshvit=${mshvit_models[$i]}
    ds_mshvit=${ds_mshvit_models[$i]}
    printf "%s %s \n"  $ds_mshvit $mshvit
    computeResult_DS_MSHViT $ds_mshvit
    computeResult_MSHViT $ds_mshvit $mshvit
done 