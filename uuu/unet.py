import torch.nn as nn
import torch

class UNet(nn.Module):
    def __init__(self, pretrained_weight=None, input_size=(512,512,1)):
        super(UNet,self).__init__()
        
        # Convolutional + Batch Normalization + ReLU layer combination define
        def CBNR(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers) #list unpacking
            
            return cbr
        
        in_channels = input_size[-1]
        
        ## Contracting path
        self.enc1_1 = CBNR(in_channels=in_channels, out_channels=64) 
        self.enc1_2 = CBNR(in_channels=64, out_channels=64) 
        self.pool1 = nn.MaxPool2d(kernel_size=2) 
        
        self.enc2_1 = CBNR(in_channels=64, out_channels=128) 
        self.enc2_2 = CBNR(in_channels=128, out_channels=128) 
        self.pool2 = nn.MaxPool2d(kernel_size=2) 
        
        self.enc3_1 = CBNR(in_channels=128, out_channels=256) 
        self.enc3_2 = CBNR(in_channels=256, out_channels=256) 
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.enc4_1 = CBNR(in_channels=256, out_channels=512)
        self.enc4_2 = CBNR(in_channels=512, out_channels=512)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.enc5_1 = CBNR(in_channels=512, out_channels=1024)
        self.enc5_2 = CBNR(in_channels=1024, out_channels=1024)
        
        ## Expansive path
        self.dec5_2 = CBNR(in_channels=1024, out_channels=512)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                         kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec4_1 = CBNR(in_channels=2*512, out_channels=512)
        self.dec4_2 = CBNR(in_channels=512, out_channels=256)
        
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                         kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec3_1 = CBNR(in_channels=2*256, out_channels=256)
        self.dec3_2 = CBNR(in_channels=256, out_channels=128)
        
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                         kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec2_1 = CBNR(in_channels=2*128, out_channels=128)
        self.dec2_2 = CBNR(in_channels=128, out_channels=64)
        
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                         kernel_size=2, stride=2, padding=0, bias=True)
        
        self.dec1_1 = CBNR(in_channels=2*64, out_channels=64)
        self.dec1_2 = CBNR(in_channels=64, out_channels=64)
        
        self.conv1x1 = nn.Sequential(
                    nn.Conv2d(in_channels=64,out_channels=1, kernel_size=1, padding=0),
                    nn.Sigmoid()
                    )
        
    def forward(self, x): # 512,512,1
        enc1_1 = self.enc1_1(x) # 512,512,64
        enc1_2 = self.enc1_2(enc1_1) # 512,512,64
        pool1 = self.pool1(enc1_2) # 256,256,64
        
        enc2_1 = self.enc2_1(pool1) # 256,256,128
        enc2_2 = self.enc2_2(enc2_1) # 256,256,128
        pool2 = self.pool2(enc2_2) # 128,128,128
        
        enc3_1 = self.enc3_1(pool2) # 128,128,256
        enc3_2 = self.enc3_2(enc3_1) # 128,128,256
        pool3 = self.pool3(enc3_2) # 64,64,256

        enc4_1 = self.enc4_1(pool3) # 64,64,512
        enc4_2 = self.enc4_2(enc4_1) # 64,64,512
        pool4 = self.pool4(enc4_2) # 32,32,512
        
        enc5_1 = self.enc5_1(pool4) # 32,32,1024
        enc5_2 = self.enc5_2(enc5_1) # 32,32,1024

        dec5_2 = self.dec5_2(enc5_2) # 32,32,512
        unpool4 = self.unpool4(dec5_2) # 64,64,512

        cat_4 = torch.cat((enc4_2,unpool4),dim=1) # 64,64,1024
        dec4_1 = self.dec4_1(cat_4) # 64,64,512
        dec4_2 = self.dec4_2(dec4_1) # 64,64,256
        unpool3 = self.unpool3(dec4_2) # 128,128,256
        
        cat_3 = torch.cat((enc3_2,unpool3),dim=1) # 128,128,512
        dec3_1 = self.dec3_1(cat_3) # 128,128,256
        dec3_2 = self.dec3_2(dec3_1) # 128,128,128
        unpool2 = self.unpool2(dec3_2) # 256,256,128
        
        cat_2 = torch.cat((enc2_2,unpool2),dim=1) # 256,256,256
        dec2_1 = self.dec2_1(cat_2) # 256,256,128
        dec2_2 = self.dec2_2(dec2_1) # 256,256,64
        unpool1 = self.unpool1(dec2_2) # 512,512,64
        
        cat_1 = torch.cat((enc1_2,unpool1),dim=1) # 512,512,128
        dec1_1 = self.dec1_1(cat_1) # 512,512,64
        dec1_2 = self.dec1_2(dec1_1) # 512,512,64

        out = self.conv1x1(dec1_2) # 512,512,1

        return out


# In[ ]:




