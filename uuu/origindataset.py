import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import random

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, mask):
        for t in self.transforms:
            seed = np.random.randint(2147483647)
            random.seed(seed)
            image = t(image)
            random.seed(seed)
            mask = t(mask)
        return image, mask
    
class OriginDataset(Dataset):
    def __init__(self,path_d,path_l,transform=None):
        self.path_data = []
        self.path_label = []
        self.transform = transform
        
        for d,l in zip(os.listdir(path_d),os.listdir(path_l)):
            path_data = os.path.join(path_d,d)
            self.path_data.append(path_data)
            
            path_label = os.path.join(path_l,l)
            self.path_label.append(path_label)

    def __len__(self):
        return len(self.path_data)
    def __getitem__(self,idx):
        train_data_path = self.path_data[idx]
        train_label_path = self.path_label[idx]
        
        train_data = Image.open(train_data_path)
        train_label = Image.open(train_label_path)

        if self.transform != None:
            train_data, train_label = self.transform(train_data, train_label)
            return train_data, train_label
       
        return train_data, train_label
