from shutil import copyfile 

if __name__ == '__main__':
    fromFile = 'results/baseline/prediction/baseline_prediction_val.csv'
    toFile =  'Resources/Dataset/SewerML/Model/ResNet-50/DS-MSHViT/Evaluation/DS-MSHViT-ResNet50_SewerML_Val.csv'
    print(toFile)
    copyfile(fromFile, toFile)