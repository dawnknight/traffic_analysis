# -*- coding: utf-8 -*-
"""
Created on Tue Feb 04 14:11:20 2014

@author: atc327
"""
import pickle
from PIL import Image
import scipy as sp
import numpy as np
import matplotlib as mpl

f = file('conv_mtx.pkl')
conv_mtx = pickle.load(f)
a = np.zeros([41,41])
for i in range(len(conv_mtx)) :
    a = conv_mtx[i]
    Max = a.max()
    Min = a.min()
    img=(a-Min)/(Max-Min)*255
    img = Image.fromarray(img.astype(np.uint8))   
    img.save('C:/Users/atc327/Desktop/Traffic data/Day/conv/conv(%.3d).png'%i)