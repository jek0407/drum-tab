import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2 as cv

class DrumDataset(Dataset):
    def __init__(self, labels, train=True, dtype='tmel'):
        label_list = np.linspace(0,22,23).astype(int)
        self.labels = labels
        self.train = train
        self.dtype = dtype
        
        if self.train == True:
            self.train_data = []
            self.train_label = []
            
            for item in label_list:
                path_t = "./dataset_train/time/"+self.labels[item]+'/'
                path_mel = "./dataset_train/mel/"+self.labels[item]+'/'
                for n,data in enumerate(os.listdir(path_t)):
                    img = self.data_config(path_t,path_mel,n,data)
                    
                    self.train_data.append(img)
                    self.train_label.append(item) 
                    del img
            self.train_data = np.array(self.train_data)
            self.train_label = np.array(self.train_label)
        
        else:
            self.valid_data = []
            self.valid_label = []
            
            for item in label_list:
                path_t = "./dataset_valid/time/"+self.labels[item]+'/'
                path_mel = "./dataset_valid/mel/"+self.labels[item]+'/'
                for n,data in enumerate(os.listdir(path_t)):   
                    img = self.data_config(path_t,path_mel,n,data)
                    
                    self.valid_data.append(img)
                    self.valid_label.append(item)
                    del img                    
            self.valid_data = np.array(self.valid_data)
            self.valid_label = np.array(self.valid_label)     
            
    def data_config(self,path_t,path_mel,n,data):
        if self.dtype == 'tmel':
            img_t = cv.imread(path_t+data,0)
            img_t = np.expand_dims(img_t, axis=2)
            img_mel = cv.imread(path_mel+os.listdir(path_mel)[n])
            img = np.concatenate((img_t,img_mel),axis=2)
            del img_t
            del img_mel
            img = np.transpose(img,(2,0,1))
            
        elif self.dtype == 'time':
            img_t = cv.imread(path_t+data,0)
            img_t = np.expand_dims(img_t, axis=2)
            img = np.transpose(img_t,(2,0,1))
            del img_t
            
        elif self.dtype == 'mel':
            img_mel = cv.imread(path_mel+os.listdir(path_mel)[n])
            img = np.transpose(img_mel,(2,0,1))
            del img_mel
            
        return img
            
            
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.valid_data)
    def __getitem__(self,idx):
        if self.train:
            return self.train_data[idx], self.train_label[idx]
        else:
            return self.valid_data[idx], self.valid_label[idx]