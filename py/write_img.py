# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 15:34:23 2014

@author: atc327
"""

import sys,pickle,glob
import numpy as np
from PIL import Image

f = file('C:/Users/atc327/Documents/GitHub/traffic_analysis/shift2.pkl')
sft = pickle.load(f)
im_roll = np.zeros([770,920,3])
path ='C:/Users/atc327/Desktop/Traffic data/Day/resize/'
imlist = glob.glob( os.path.join(path, '*.png') )
UL = [20,20]
LR = [750,900]
#UL = [80,80]
#LR = [121,121]

for i in range(len(imlist)):
    print i
    y,x = sft[i]
    im_roll[10:760,10:910,:] = np.array(Image.open(imlist[i])).astype(np.float)
    
    im_roll = np.roll(np.roll(im_roll,y,axis=0),x,axis=1)    
    
          
   
    im_cut = im_roll[UL[0]:LR[0],UL[1]:LR[1]]
    im = Image.fromarray(im_cut.astype(np.uint8))   
    im.save('C:/Users/atc327/Desktop/Traffic data/Day/reg_2/%.3d.png'%i)
