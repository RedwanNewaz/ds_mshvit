modelPath='results/joint_vit/SewerMLJoint_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn/version_32'
outputDir="Resources/Dataset/SewerML/Model/ResNet-50/DS-MSHViT/Weight"
modelName='DS-MSHViT-ResNet-50'

PY_FILE="/home/aredwann/PyDev/SewerML-tracker/metrics/extract_weights.py"
PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"

CUDA_VISIBLE_DEVICES=2  $PYTHON $PY_FILE  --model_path $modelPath \
                  --output_dir $outputDir \
                  --model_name $modelName