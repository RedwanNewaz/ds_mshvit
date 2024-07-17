# version='32'
# modelPath="/home/aredwann/PyDev/SewerML-tracker/results/joint_vit/SewerMLJoint_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn/version_$version"
# outputDir="/home/aredwann/PyDev/SewerML-tracker/results/joint_vit/weights"
# modelName="SewerML_RGB_resnet50_MultiScaleViTHead_FNetBlock_Sinkhorn"

# PY_FILE="/home/aredwann/PyDev/SewerML-tracker/metrics/seperate_rgb_weights.py"
# PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"

# CUDA_VISIBLE_DEVICES=2  $PYTHON $PY_FILE  --model_path $modelPath \
#                   --output_dir $outputDir \
#                   --model_name $modelName



rawModels=(
    "results/joint_vit/SewerMLJoint_resnet18_MultiScaleViTHead_FNetBlock_Sinkhorn/version_82"
    "results/joint_vit/SewerMLJoint_resnet34_MultiScaleViTHead_FNetBlock_Sinkhorn/version_83"
)

outModels=(
    "ResNet-18"
    "ResNet-34"
)

for i in {0..2}; do
  # Code to be executed for each value of i
  modelPath=${rawModels[i]}
  outputDir="Resources/Dataset/SewerML/Model/${outModels[i]}/DS-MSHViT/Weight"
  modelName="DS-MSHViT-${outModels[i]}-RGB"
  echo $modelPath
  echo $outputDir
  echo $modelName
  PY_FILE="/home/aredwann/PyDev/SewerML-tracker/metrics/seperate_rgb_weights.py"
  PYTHON="/home/aredwann/anaconda3/envs/SewerML-tracker/bin/python"
  CUDA_VISIBLE_DEVICES=2  $PYTHON $PY_FILE  --model_path $modelPath \
                    --output_dir $outputDir \
                    --model_name $modelName
done