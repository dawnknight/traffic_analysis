# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 15:34:23 2014

@author: atc327
"""

import sys,pickle,glob
import numpy as np
from PIL import Image

f = file('sft.pkl')
sft = pickle.load(f)
im = np.zeros([700,550,3])
path ='C:/Users/atc327/Desktop/Traffic data/Day/resize/'
imlist = glob.glob( os.path.join(path, '*.png') )
UL = [30,30]
LR = [730,930]

for i in range(len(imlist)):
    print i
    im = np.array(Image.open(imlist[i])).astype(np.float)
    y,x = sft[i]
          
    im = np.roll(im,-y,axis=0)
    im = np.roll(im,-x,axis=1)      
    im = im[UL[0]:LR[0],UL[1]:LR[1],0:]
    im = Image.fromarray(im.astype(np.uint8))   
    im.save('C:/Users/atc327/Desktop/Traffic data/Day/reg_m/out(%.3d).png'%i)
