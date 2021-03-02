from pathlib import Path
from math import log10
import argparse
from pathlib import Path
import random
import time
import os
import subprocess
import subprocess
import threading
import queue

import torch
import torch.utils.data as data
from torch import nn
from torch.nn.functional import relu
from torchvision import transforms
from torchvision.transforms import ToTensor, RandomCrop
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision 

import cv2
import numpy as np
import kornia
import d3dshot
import win32gui, win32ui, win32con, win32api

torch.backends.cudnn.benchmark = True
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(dev)

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv0   = nn.Conv2d(1, 16, 3, padding=1, padding_mode='replicate' )

        self.LR1_1   = nn.LeakyReLU()
        self.conv1_1 = nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate')
        self.LR1_2   = nn.LeakyReLU()
        self.DO1     = nn.Dropout(0.3)
        self.conv1_2 = nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate')


        self.LR2_1   = nn.LeakyReLU()
        self.conv2_1 = nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate')
        self.LR2_2   = nn.LeakyReLU()
        self.DO2     = nn.Dropout(0.3)
        self.conv2_2 = nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate')
        
        
        self.LR3_1   = nn.LeakyReLU()
        self.conv3_1 = nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate')
        self.LR3_2   = nn.LeakyReLU()
        self.DO3     = nn.Dropout(0.3)
        self.conv3_2 = nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate')
        

        self.convL   = nn.Conv2d(16, 1, 3, padding=1, padding_mode='replicate')
    def forward(self, x):

        A0 = self.conv0(x)

        x = self.LR1_1(A0)      
        x = self.conv1_1(x)
        x = self.LR1_2(x)
        x = self.DO1(x)
        x = self.conv1_2(x)
        x=x+A0
        A1=x

        x = self.LR2_1(x)      
        x = self.conv2_1(x)
        x = self.LR2_2(x)
        x = self.DO2(x)
        x = self.conv2_2(x)
        x=x+A0+A1
        A2=x        
        
        x = self.LR3_1(x)      
        x = self.conv3_1(x)
        x = self.LR3_2(x)
        x = self.DO3(x)
        x = self.conv3_2(x)
        x=x+A0+A1+A2

        x = self.convL(x)
        return x

model = SRCNN()
model_path = 'r3c1x240_D+vgg.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device(dev)))
model=model.to(dev).half()
model=torch.jit.script(model)



d = d3dshot.create(capture_output="numpy")
d.display = d.displays[1]
fps=30

fB = queue.Queue(maxsize=3)



def ss():
    while True: 
        start_time = time.perf_counter()
        
        img=d.screenshot()   
        frame = img
        fB.put(frame)
        
        x1=time.perf_counter() - start_time 
        start_timeD = time.perf_counter()
        print(x1)
        while (time.perf_counter() - start_timeD) < (1/fps) - x1 :
            pass
    
def Pred():
       
  cv2.namedWindow("cnn")  
  a = win32gui.FindWindow(None,"cnn")
  win32gui.SetWindowLong(a, win32con.GWL_STYLE, win32con.WS_POPUP)
  x = 0
  y = 0
  width = 1920
  height = 1080 
    
    
  while True:
    
    start_time = time.perf_counter()
    
    
    img=fB.get(timeout=2)
    frame = img
    frame= torch.from_numpy(frame).to(dev).reshape([1, 1080, 1920, 3]).permute(0, 3, 1, 2).float()#nchw
    
    frameYUV= kornia.rgb_to_yuv(frame)
    frameY  = frameYUV[:,0,:,:].reshape([1, 1, 1080, 1920]).half()
    predY=model(frameY/255)*255 

    f0=predY.float()[0,0,:,:]
    f1=frameYUV[0,1,:,:]
    f2=frameYUV[0,2,:,:] #1080,1920
    pred=torch.stack([f0,f1,f2]).float() # 3,1080,1920
    pred=kornia.yuv_to_rgb(pred).clip(0,255).permute(1,2,0)[:,:,[2, 1, 0]]
    pred=torch.tensor(pred,dtype=torch.uint8).to('cpu').numpy()

    cv2.imshow('cnn', pred)
    win32gui.SetWindowPos(a, win32con.HWND_TOPMOST, x, y, width, height, win32con.SWP_SHOWWINDOW)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break  
    
    x1=time.perf_counter() - start_time 
    start_timeD = time.perf_counter()



T_SS = threading.Thread(target=ss)
T_Pred = threading.Thread(target=Pred)

T_SS.start()
T_Pred.start()
