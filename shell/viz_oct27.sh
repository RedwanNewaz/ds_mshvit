export CUDA_HOME=~/anaconda3/envs/SewerML-tracker3
ROOT="/home/aredwann/PyDev/SewerML-tracker"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/visualize_inference.py"
DATASET="SewerML"
DEVICE=3

computeResult_DS_MSHViT()
{
    model=$1
    ann_root="$ROOT/Resources/Dataset/Carencro/Annotation/Test"
    data_root="$ROOT/Resources/Dataset/${DATASET}/Data/Validation/val_data"
    model_path="${ROOT}/Resources/Dataset/SewerML/Model/${model}/DS-MSHViT/Weight/DS-MSHViT-${model}-RGB.ckpt"
    results_output="${ROOT}/Resources/Dataset/Carencro/Model/${model}/DS-MSHViT/Prediction"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON $PY_FILE --model_path $model_path \
                    --model_version DS-MSHViT-${model}-Viz_Test \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_test 
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

computeResult_DS_MSHViT ResNet-101

# computeResult_MSHViT ResNet-101 resnet101
