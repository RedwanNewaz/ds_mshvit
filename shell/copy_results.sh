

DATASET="Carencro"

NETS=(
"CoAtNet-0"
"CoAtNet-1"
"BotNet-50-S1"
"ResNet-50"
"ResNet-101"
"TResNet-L"
"TResNet-M"
)

NETWORKS=(
"coatnet0"
"coatnet1"
"botnet"
"resnet50"
"resnet101"
"tresnetl"
"tresnetm"
)

# Resources/Dataset/Carencro/Model/ResNet-50/MSHViT

batch_cp_pred()
{
    FOLDER="MSHViT"
    for i in {0..6}
    do 
        NET=${NETS[i]}
        NETWORK=${NETWORKS[i]}_mshvit

        TO_RES="Resources/Dataset/${DATASET}/Model/${NET}/${FOLDER}/Prediction/${FOLDER}-${NET}_${DATASET}_Val.csv"
        FROM_RES="results/rgb_vit/${NETWORK}/benchmark_prediction_val.csv"
        # cp $FROM_RES $TO_RES
        printf "%s -> %s \n" $FROM_RES $TO_RES
    done 
}

