import os
import h5py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class build_Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None, model="None"):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.model = model

        if self.split == "train_Pancreas":
            labeled_path = os.path.join(self.data_dir + "/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/Pancreas_h5/" + image_name + "_norm.h5" for image_name in self.image_list]
            print("train_Pancreas total {} samples".format(len(self.sample_list)))
        elif self.split == "train_BraTS2019":
            labeled_path = os.path.join(self.data_dir + "/data/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            print("train_BraTS2019 total {} samples".format(len(self.sample_list)))
        elif self.split == "train_Lung":
            labeled_path = os.path.join(self.data_dir + "/train.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n', '') for item in self.image_list]
            self.sample_list = [self.data_dir + "/lung_h5/" + image_name + ".h5" for image_name in self.image_list]
            print("train_BraTS2019 total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(case, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample
