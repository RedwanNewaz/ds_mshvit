import pandas as pd
import os
import numpy as np
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool
CHOSEN_LABELS = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]

dataDict = {}

def saveDataAndLabels(item):
    df, className, args = item
    print("=" * 50, className, "=" * 50)
    # create a folder to symbolic link these images
    out_img_dir = os.path.join(args.output_dir, args.folder_type, className)
    os.makedirs(out_img_dir, exist_ok=True)

    for jitem in df["Filename"]:
        img_file = dataDict[jitem]
        os.system("ln -s %s %s" % (img_file, out_img_dir))
        # print(img_file)
    # save a csv file
    outfile = os.path.join(out_img_dir, className + "_" + os.path.basename(args.csv_file))
    df.to_csv(outfile)

def seperateClass(args):
    df = pd.read_csv(args.csv_file)
    os.makedirs(args.output_dir, exist_ok=True)
    header = df.columns

    for className in header[1:]:
        sep = df[df[className] == 1] # looking for a particular label
        if className not in CHOSEN_LABELS:
            continue
        # sample image paths
        sample_size = min(len(sep), args.sample_size)
        sep = sep.sample(n=sample_size)
        print(f"[+] total {className} labels = {len(sep)}" )
        yield  (sep, className, args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--csv-file", type=str, default="/usace_share/SewerML/SewerML_Train.csv")
    parser.add_argument("--sample-size", type=int, default=2000)
    parser.add_argument("--folder-type", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="/usace_share/SewerML/SampleDataset")


    args = parser.parse_args()

    print("[+] loading data path ...")

    for item in Path(os.path.dirname(args.csv_file)).glob("%s*" % args.folder_type):
        if "zip" in item.name:
            continue
        for img in Path(item).glob("*.png"):
            dataDict[img.name] = img
    print("[+] loading complete")

    tasks = seperateClass(args)
    with Pool() as pool:
        pool.map(saveDataAndLabels, tasks)


