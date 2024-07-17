#!/home/aredwann/anaconda3/envs/MSHViT/bin/python
import os
import json
import argparse
import pandas as pd
import numpy as np
from multilable_metrics import multilabel_sewerml_evaluation
            

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def sewerml_results(scoresDf, targetPath, split):
    
    # A priori known label names and weights for eval.
    LabelWeightDict = {"RB":5.55,"OB":3.0625,"PF":1.6075,"DE":0.9,"FS":3.5625,"IS":1.025,"RO":1.975,"IN":1.7375,"AF":0.45,"BE":1.2625,"FO":1.375,"GR":0.5,"PH":2.3125,"PB":2.3125,"OS":5.0,"OP":2.125,"OK":2.44}
    Labels = list(LabelWeightDict.keys())
    LabelWeights = list(LabelWeightDict.values())


    # Load data from csv files
    targetSplitpath = os.path.join(targetPath, "SewerML_{}.csv".format(split))
    targetsDf = pd.read_csv(targetSplitpath, sep=",", encoding="utf-8")
    targetsDf = targetsDf.sort_values(by=["Filename"]).reset_index(drop=True)

        
    scores = scoresDf[Labels].values
    targets = targetsDf[Labels].values
    


    new, main, auxillary = multilabel_sewerml_evaluation(scores, targets, LabelWeights)

    resultsDict = {"Labels": Labels, "LabelWeights": LabelWeights, "New": new, "Main": main, "Auxillary": auxillary}

    resultsStr = ""
    resultsStr += "New metrics: " + "{:.2f} & {:.2f} ".format(new["F2"]*100,  auxillary["F1_class"][-1]*100) + "\n"
    resultsStr += "ML main metrics: " + "{:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f}".format(main["mF1"]*100, main["MF1"]*100, main["OF1"]*100, main["OP"]*100, main["OR"]*100, main["CF1"]*100, main["CP"]*100, main["CR"]*100, main["EMAcc"]*100, main["mAP"]*100) + "\n"
    resultsStr += "Class F1: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["F1_class"]]) + "\n"
    resultsStr += "Class F2: " + " & ".join(["{:.2f}".format(x*100) for x in new["F2_class"]]) + "\n"
    resultsStr += "Class Precision: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["P_class"]]) + "\n"
    resultsStr += "Class Recall: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["R_class"]]) + "\n"
    resultsStr += "Class AP: " + " & ".join(["{:.2f}".format(x*100) for x in auxillary["AP"]]) + "\n"

    return resultsDict, resultsStr

    
def calcualteResults(args):
    scorePath = args["score_path"]
    targetPath = args["gt_path"]
    outputPath = args["output_path"]
    dataset = args["dataset"]

    os.makedirs(outputPath, exist_ok=True)

    split = args["split"]


    for subdir, dirs, files in os.walk(scorePath):
        print(subdir)
        for scoreFile in files:
            if split.lower() not in scoreFile:
                continue
            if  "defect" not in scoreFile and "prediction" not in scoreFile:
                continue
            if os.path.splitext(scoreFile)[-1] != ".csv":
                continue
            print(scoreFile)
            
            scoresDf = pd.read_csv(os.path.join(subdir, scoreFile), sep=",")
            scoresDf = scoresDf.sort_values(by=["Filename"]).reset_index(drop=True)

            print(scoresDf)

            if dataset == "SewerML":
                resultsDict, resultsStr = sewerml_results(scoresDf, targetPath, split)

            outputName = "{}_{}".format(split, scoreFile)
            if split.lower() == "test":
                outputName = outputName[:len(outputName) - len("_test.csv")]
            elif split.lower() == "val":
                outputName = outputName[:len(outputName) - len("_val.csv")]
            elif split.lower() == "train":
                outputName = outputName[:len(outputName) - len("_train.csv")]


            with open(os.path.join(outputPath,'{}.json'.format(outputName)), 'w') as fp:
                json.dump(resultsDict, fp, cls=NumpyEncoder, indent=4)

            with open(os.path.join(outputPath,'{}_latex.txt'.format(outputName)), "w") as text_file:
                text_file.write(resultsStr)
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default = "/home/aredwann/PyDev/SewerML-tracker/results/baseline")
    parser.add_argument("--split", type=str, default = "Val", choices=["Train", "Val", "Test"])
    parser.add_argument("--score_path", type=str, default = "./results")
    parser.add_argument("--gt_path", type=str, default = "./annotations_sewerml")
    parser.add_argument('--dataset', type=str, default ="SewerML", choices=["SewerML"])

    args = vars(parser.parse_args())
    # CUDA_VISIBLE_DEVICES=2 
    calcualteResults(args)
