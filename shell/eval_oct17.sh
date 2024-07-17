export CUDA_HOME=~/anaconda3/envs/SewerML-tracker3
ROOT="/home/aredwann/PyDev/SewerML-tracker"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/baseline_inference.py"
DATASET="Carencro"
DEVICE=2

computeResult_DS_MSHViT()
{
    model=$1
    ann_root="$ROOT/Resources/Dataset/${DATASET}/Annotation/Validation"
    data_root="$ROOT/Resources/Dataset/${DATASET}/Data/Validation/val_data"
    model_path="${ROOT}/Resources/Dataset/SewerML/Model/${model}/DS-MSHViT/Weight/DS-MSHViT-${model}-RGB.ckpt"
    results_output="${ROOT}/Resources/Dataset/${DATASET}/Model/${model}/DS-MSHViT/Prediction"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON $PY_FILE --model_path $model_path \
                    --model_version DS-MSHViT-${model}-RGB_Val.csv \
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
                    --model_version MSHViT-${model}_${DATASET}_Val.csv \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_val 
    echo "[+] computing results for $1 in CUDA $DEVICE is finished !!!"
}

TEST_DS_MSHViT_SewerML()
{
    model=$1
    ann_root="$ROOT/Resources/Dataset/${DATASET}/Annotation/Test"
    data_root="$ROOT/Resources/Dataset/${DATASET}/Data/Test/test_data"
    model_path="${ROOT}/Resources/Dataset/SewerML/Model/${model}/DS-MSHViT/Weight/DS-MSHViT-${model}-RGB.ckpt"
    results_output="${ROOT}/Resources/Dataset/${DATASET}/Model/${model}/DS-MSHViT/Prediction"

    CUDA_VISIBLE_DEVICES=$DEVICE $PYTHON $PY_FILE --model_path $model_path \
                    --model_version DS-MSHViT-${model}-RGB_Test \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_test 
    echo "[+] computing results for $1 in CUDA $DEVICE is finished !!!"
}



# weights=(
#     "BotNet-50-S1"
#     "CoAtNet-0"
#     "ResNet-18"
#     "ResNet-34"
#     "ResNet-50"
#     "ResNet-101"
#     "TResNet-M"
#     "TResNet-L"
# )

nets=(
    "ResNet-18"
    "ResNet-34"
)

weights=(
    "resnet18"
    "resnet34"
)

BatchPredDS_MSHViT()
{
    for net in ${weights[@]}
    do 
        printf "[+] Evaluating $net ...\n"
        computeResult_DS_MSHViT $net
    done 
}

BatchPredCarncroDataset()
{
    for net in ${weights[@]}
    do 
        
        results_output="${ROOT}/Resources/Dataset/${DATASET}/Model/${net}/DS-MSHViT/Prediction"
        TARGET_FILE="${results_output}/DS-MSHViT-${net}-RGB_Test.csv"
        if [ -f "$TARGET_FILE" ]
        then
            # echo "$TARGET_FILE exists."
            echo ""
        else
            printf "[+] Evaluating $net ...\n"
            TEST_DS_MSHViT_SewerML $net
        fi
    done 
}

for i in 0 1 
do 
    echo "${nets[$i]}" "${weights[$i]}" 
    computeResult_MSHViT "${nets[$i]}" "${weights[$i]}" 
done



# Resources/Dataset/SewerML/Model/ResNet-34/MSHViT/Weight/resnet18_mshvit.ckpt