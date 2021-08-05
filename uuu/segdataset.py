import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
import numpy as np
import random
    
class SegDataset(Dataset):
    def __init__(self,path_aug,transform=None):
        self.path_data = []
        self.path_label = []
        self.transform = transform
        
        for path_data,path_label in zip(glob(r'./data/train/aug/*_img.png'),glob(r'./data/train/aug/*_mask.png')):
            self.path_data.append(path_data)
            self.path_label.append(path_label)

    def __len__(self):
        return len(self.path_data)
    def __getitem__(self,idx):
        train_data_path = self.path_data[idx]
        train_label_path = self.path_label[idx]
        
        train_data = Image.open(train_data_path)
        train_label = Image.open(train_label_path)
        
        train_data = self.transform(train_data)
        train_label = self.transform(train_label)
       
        return train_data, train_label

