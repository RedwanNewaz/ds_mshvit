
model_path="/home/aredwann/MSHViT/weights/resnet50_mshvit.ckpt"
model_version="baseline"
results_output="/home/aredwann/PyDev/SewerML-tracker/results/baseline/prediction"
ann_root="/home/aredwann/PyDev/SewerML-tracker/annotations_sewerml"
data_root="/home/aredwann/MSHViT/dataLink/rgb_val"

# mkdir $results_output

PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
PY_FILE="/home/aredwann/PyDev/SewerML-tracker/baseline_inference.py"


# $PYTHON $PY_FILE --model_path $model_path \
#                 --results_output $results_output \
#                 --model_version $model_version \
#                 --data_root $data_root\
#                 --ann_root $ann_root \
#                 --dataset SewerML \
#                 --batch_size 128 \
#                 --eval_val


PY_FILE="../metrics/calculate_results.py"

$PYTHON $PY_FILE --score_path $results_output --split Val --gt_path $ann_root