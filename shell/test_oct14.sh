export CUDA_HOME=~/anaconda3/envs/SewerML-tracker3

ROOT="/home/aredwann/PyDev/SewerML-tracker"
ann_root="$ROOT/Resources/Dataset/SewerML/Annotation/Test"
data_root="$ROOT/Resources/Dataset/SewerML/Data/Test/test_data"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/baseline_inference.py"

computeResult()
{
    echo "[+] computing results for $2 in CUDA $4"
    model_path="$1/$2.ckpt"
    model_name="${2}_Test"
    results_output=$3


    CUDA_VISIBLE_DEVICES=$4 $PYTHON $PY_FILE --model_path $model_path \
                    --model_version $model_name \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_test 

}


OUT_DS_MSHVIT_DIRS_PART1=(
"BotNet-50-S1"
"CoAtNet-0"
"ResNet-34"
"ResNet-101"
)

OUT_DS_MSHVIT_DIRS_PART2=(
"TResNet-M"
"TResNet-L"
)

OUT_DS_MSHVIT_DIRS_PART3=(
"ResNet-18"
"ResNet-34"
)

DEVICE=1

for var in ${OUT_DS_MSHVIT_DIRS_PART3[@]}
do
    model_path="Resources/Dataset/SewerML/Model/${var}/DS-MSHViT/Weight"
    model_name="DS-MSHViT-${var}-RGB"
    results_output="Resources/Dataset/SewerML/Model/${var}/DS-MSHViT/Prediction"

    printf " [ModelPath]: %s \n [ModelName]: %s \n [OutputDir]: %s \n" $model_path $model_name $results_output
    computeResult $model_path $model_name $results_output $DEVICE
done 