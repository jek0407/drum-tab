import torch
from torch.utils.data import Dataset
import numpy as np
import os, glob
import cv2 as cv
import natsort

class DrumDataset_test(Dataset): #dtype = 'tmel', 'time', 'mel
    def __init__(self,labels,path_mel):
        
        self.path_mel = path_mel
        self.labels = labels
        self.test_data = []
        self.test_label = [] 
        
        
        for path_mel_name in natsort.natsorted(glob.glob(os.path.join(path_mel,'*'))):
            label = 'X'
            # print(path_mel_name)
            img_mel = cv.imread(path_mel_name)
            img = np.transpose(img_mel,(2,0,1))

            self.test_data.append(img)
            self.test_label.append(label) 
        
    def __len__(self):
        return len(self.test_data)
    
    def __getitem__(self,idx):
        self.test_data = np.array(self.test_data)
        self.test_label = np.array(self.test_label)
        return self.test_data[idx], self.test_label[idx]