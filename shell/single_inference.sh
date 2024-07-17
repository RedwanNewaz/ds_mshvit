version='32'
ROOT="/home/aredwann/PyDev/SewerML-tracker"
ann_root="$ROOT/annotations_sewerml/Johny_091023"
data_root="$ROOT/annotations_sewerml/Johny_old/data"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/baseline_inference.py"
METRIC_PY_FILE="$ROOT/metrics/calculate_results.py"
dev_model_weight="$ROOT/results/joint_vit/weights/SewerML_RGB_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn_RGB_version_$version.ckpt"
baseline_model_weight="/home/aredwann/MSHViT/weights/resnet50_hvit_sinkhorn.ckpt"


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




# computeResult $dev_model_weight "dev_091023" 2

computeResult $baseline_model_weight "baseline_091023" 1

