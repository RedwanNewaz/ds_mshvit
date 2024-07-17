import os
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from torchvision import datasets as datasets

ML_Datasets = ["SewerML", "SewerMLJoint", "SewerMLSequence"]

class SewerMLDataset(Dataset):
    def __init__(self, annRoot, imgRoot, split="Test", transform=None, loader=default_loader):
        super(SewerMLDataset, self).__init__()
        self.imgRoot = imgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = ["RB","OB","PF","DE","FS","IS","RO","IN","AF","BE","FO","GR","PH","PB","OS","OP","OK"]
        self.num_classes = len(self.LabelNames)

        # self.loadAnnotations()
        if split == "Test":
            self.test_loadAnnotations()
        else:
            self.loadAnnotations()


    def test_loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        print(gtPath)
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename"])
        self.imgPaths = gt["Filename"].values
        labels =['WaterLevel','VA','RB','OB','PF','DE','FS','IS','RO','IN','AF','BE','FO','GR','PH','PB','OS','OP','OK','ND','Defect']
        M = len(self.imgPaths )
        N =  len(labels)
        self.labels = np.ones((M, N), dtype=int)


    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        print(gtPath)
        # gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = ["Filename"])
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols = self.LabelNames + ["Filename"])
        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values
        # labels =['WaterLevel','VA','RB','OB','PF','DE','FS','IS','RO','IN','AF','BE','FO','GR','PH','PB','OS','OP','OK','ND','Defect']

        # M = len(self.imgPaths )
        # N =  len(labels)
        # self.labels = np.ones((M, N), dtype=int)
        # # self.labels = np.expand_dims(self.labels, axis=1)
    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(self.labels[index, :], dtype=torch.float)

        return img, target, path

class SewerMLSequenceDataset(SewerMLDataset):

    def getSingleItem(self, index):
        index = min(index, len(self.imgPaths) - 1)

        path = self.imgPaths[index]

        img = self.loader(os.path.join(self.imgRoot, path))
        if self.transform is not None:
            img = self.transform(img)

        target = torch.tensor(self.labels[index, :], dtype=torch.float)

        return img, target, path

    def __getitem__(self, index):
        img1, target1, path1 = self.getSingleItem(index)
        img2, target2, path2 = self.getSingleItem(index + 1)

        return (img1, img2), (target1, target2), (path1, path2)

def get_dataset(dataset_name, ann_root, data_root, split, transform):

    if dataset_name == "SewerML":
        dataset = SewerMLDataset(ann_root, data_root, split=split, transform=transform)
    elif dataset_name == "SewerMLSequence":
        dataset = SewerMLSequenceDataset(ann_root, data_root, split=split, transform=transform)
    else:
        raise ValueError("There are no Dataset for the supplied dataset: {}".format(dataset_name))
    
    return dataset
        
def get_dataloader(dataset_name, batch_size, workers, ann_root, data_root, split, transform):

    dataset = get_dataset(dataset_name, ann_root, data_root, split, transform)   
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers = workers, pin_memory=True)
    
    return dataloader, dataset.LabelNames


class SewerMLJointDataset(Dataset):
    def __init__(self, annRoot, rgbImgRoot, optiImgRoot, split="Test", transform=None, loader=default_loader):
        super(SewerMLJointDataset, self).__init__()
        self.rgbImgRoot = rgbImgRoot
        self.optiImgRoot = optiImgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS",
                           "OP", "OK"]
        self.num_classes = len(self.LabelNames)

        if split == "Test":
            self.test_loadAnnotations()
        else:
            self.loadAnnotations()

    def test_loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        print(gtPath)
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=["Filename"])
        self.imgPaths = gt["Filename"].values
        labels = ['WaterLevel', 'VA', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH',
                  'PB', 'OS', 'OP', 'OK', 'ND', 'Defect']
        M = len(self.imgPaths)
        N = len(labels)
        self.labels = np.ones((M, N), dtype=int)

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=self.LabelNames + ["Filename"])
        self.imgPaths = gt["Filename"].values
        self.labels = gt[self.LabelNames].values

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        rgb_img = self.loader(os.path.join(self.rgbImgRoot, path))
        opti_img = self.loader(os.path.join(self.optiImgRoot, path))
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            opti_img = self.transform(opti_img)


        target = torch.tensor(self.labels[index, :], dtype=torch.float)

        return rgb_img, opti_img, target, path

class SewerMLJointDatasetV2(Dataset):
    def __init__(self, annRoot, rgbImgRoot, optiImgRoot, split="Test", transform=None, loader=default_loader):
        super(SewerMLJointDatasetV2, self).__init__()
        self.rgbImgRoot = rgbImgRoot
        self.optiImgRoot = optiImgRoot
        self.annRoot = annRoot
        self.split = split

        self.transform = transform
        self.loader = loader

        self.LabelNames = ["RB", "OB", "PF", "DE", "FS", "IS", "RO", "IN", "AF", "BE", "FO", "GR", "PH", "PB", "OS",
                           "OP", "OK"]
        self.num_classes = len(self.LabelNames)

        if split == "Test":
            self.test_loadAnnotations()
        else:
            self.loadAnnotations()

    def test_loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        print(gtPath)
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=["Filename"])
        self.imgPaths = gt["Filename"].values
        labels = ['WaterLevel', 'VA', 'RB', 'OB', 'PF', 'DE', 'FS', 'IS', 'RO', 'IN', 'AF', 'BE', 'FO', 'GR', 'PH',
                  'PB', 'OS', 'OP', 'OK', 'ND', 'Defect']
        M = len(self.imgPaths)
        N = len(labels)
        self.labels = np.ones((M, N), dtype=int)

    def loadAnnotations(self):
        gtPath = os.path.join(self.annRoot, "SewerML_{}.csv".format(self.split))
        gt = pd.read_csv(gtPath, sep=",", encoding="utf-8", usecols=self.LabelNames + ["Filename"])
        self.imgPaths = np.array(["%04d.png" % item for item in gt["Filename"][1:].values])
        self.labels = gt[self.LabelNames].values

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        path = self.imgPaths[index]

        # print("[+RGB_DATA] loading from: ", self.rgbImgRoot)
        # print("[+Opti_DATA] loading from: ", self.optiImgRoot)

        rgb_img = self.loader(os.path.join(self.rgbImgRoot, path))
        opti_img = self.loader(os.path.join(self.optiImgRoot, path))
    

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            opti_img = self.transform(opti_img)


        target = torch.tensor(self.labels[index, :], dtype=torch.float)

        return rgb_img, opti_img, target, path


def get_joint_dataloader(dataset_name, batch_size, workers, ann_root, rgb_data_root, opti_data_root, split, transform):
    dataset = SewerMLJointDataset(ann_root,  rgb_data_root, opti_data_root, split, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True)

    return dataloader, dataset.LabelNames


    