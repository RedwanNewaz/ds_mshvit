version='32'
ROOT="/home/aredwann/PyDev/SewerML-tracker"
ann_root="$ROOT/annotations_sewerml"
data_root="$ROOT/annotations_sewerml/test_data"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="$ROOT/baseline_inference.py"
METRIC_PY_FILE="$ROOT/metrics/calculate_results.py"
LINK_PY_FILE="$ROOT/utility/SymlinkImages.py"
dev_model_weight="$ROOT/results/joint_vit/weights/SewerML_RGB_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn_RGB_version_$version.ckpt"
baseline_model_weight="/home/aredwann/MSHViT/weights/resnet50_hvit_sinkhorn.ckpt"


computeResult()
{
    model_path=$1
    results_output="$ROOT/results/rgb_vit_sewer_ml_test/$2"
    mkdir -p $results_output

    CUDA_VISIBLE_DEVICES=$3 $PYTHON $PY_FILE --model_path $model_path \
                    --model_version "benchmark" \
                    --results_output $results_output \
                    --data_root $data_root\
                    --ann_root $ann_root \
                    --eval_test

    # $PYTHON $METRIC_PY_FILE --score_path $results_output --split Val --gt_path $ann_root --output_path $results_output
    # computeResultNormalized $1 $2
}


# https://competitions.codalab.org/competitions/32705#learn_the_details-submission

computeResult $dev_model_weight "dev" 0

# computeResult $baseline_model_weight "baseline" 1
# SRC="/usace_share/SewerML/ttest01"
# DEST="/home/aredwann/PyDev/SewerML-tracker/annotations_sewerml/test_data"
# $PYTHON $LINK_PY_FILE --src-dir $SRC --dest-dir $DEST

