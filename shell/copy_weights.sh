backbone_weights=(
botnet.ckpt
coatnet0.ckpt
coatnet1.ckpt
resnet101.ckpt
resnet18.ckpt
resnet34.ckpt
resnet50.ckpt
tresnetl.ckpt
tresnetm.ckpt
)

mshvit_weight=(
botnet_mshvit.ckpt
coatnet0_mshvit.ckpt
coatnet1_mshvit.ckpt
resnet101_mshvit.ckpt
resnet18_mshvit.ckpt
resnet34_mshvit.ckpt
resnet50_mshvit.ckpt
tresnetl_mshvit.ckpt
tresnetm_mshvit.ckpt 
)

CP_ROOT="/home/aredwann/MSHViT/weights"

OUT_BACKBONE_DIRS=(
Resources/Dataset/SewerML/Model/BotNet-50-S1/Backbone/Weight
Resources/Dataset/SewerML/Model/CoAtNet-0/Backbone/Weight
Resources/Dataset/SewerML/Model/CoAtNet-1/Backbone/Weight
Resources/Dataset/SewerML/Model/ResNet-18/Backbone/Weight
Resources/Dataset/SewerML/Model/ResNet-34/Backbone/Weight
Resources/Dataset/SewerML/Model/ResNet-50/Backbone/Weight
Resources/Dataset/SewerML/Model/ResNet-101/Backbone/Weight
Resources/Dataset/SewerML/Model/TResNet-M/Backbone/Weight
Resources/Dataset/SewerML/Model/TResNet-L/Backbone/Weight
)

OUT_MSHVIT_DIRS=(
Resources/Dataset/SewerML/Model/BotNet-50-S1/MSHViT/Weight
Resources/Dataset/SewerML/Model/CoAtNet-0/MSHViT/Weight
Resources/Dataset/SewerML/Model/CoAtNet-1/MSHViT/Weight
Resources/Dataset/SewerML/Model/ResNet-18/MSHViT/Weight
Resources/Dataset/SewerML/Model/ResNet-34/MSHViT/Weight
Resources/Dataset/SewerML/Model/ResNet-50/MSHViT/Weight
Resources/Dataset/SewerML/Model/ResNet-101/MSHViT/Weight
Resources/Dataset/SewerML/Model/TResNet-M/MSHViT/Weight
Resources/Dataset/SewerML/Model/TResNet-L/MSHViT/Weight
)

for i in {0..8}
do 
    FROM_BACKBONE_DIR="$CP_ROOT/${backbone_weights[i]}"
    TO_BACKBONE_DIR="${OUT_BACKBONE_DIRS[i]}/${backbone_weights[i]}"

    FROM_MSHVIT_DIR="$CP_ROOT/${mshvit_weight[i]}"
    TO_MSHVIT_DIR="${OUT_MSHVIT_DIRS[i]}/${mshvit_weight[i]}"

    # printf "%s -> %s \n" $FROM_BACKBONE_DIR $TO_BACKBONE_DIR
    printf "%s -> %s \n" $FROM_MSHVIT_DIR $TO_MSHVIT_DIR
    cp $FROM_BACKBONE_DIR $TO_BACKBONE_DIR
    cp $FROM_MSHVIT_DIR $TO_MSHVIT_DIR
done 