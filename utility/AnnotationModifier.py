from pathlib import Path 
import pandas as pd 
from argparse import ArgumentParser
import os 


def modifyAnnotation(args):

    df = pd.read_csv(args.anno_csv)
    filter_list = [img.name for img in sorted(Path(args.data_dir).glob('*.png'))]
    filtered_df = df[df['Filename'].isin(filter_list)]

    print(filtered_df)
    filtered_df.to_csv(args.out_csv, index=False)
    print("[+] file saved ", args.out_csv)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/home/aredwann/PyDev/SewerML-tracker/dataset/train', help='directory of image files')
    parser.add_argument('--anno-csv', type=str, default='/home/aredwann/MSHViT/annotations_sewerml/ori/SewerML_Valid.csv', help='original annoation csv file')
    parser.add_argument('--out-csv', type=str, default='../annotations_sewerml/SewerML_Train_sampled.csv', help='output annotation csv file')

    args = parser.parse_args()
    modifyAnnotation(args)
