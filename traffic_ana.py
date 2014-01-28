# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:08:00 2014

@author: atc327
"""
import os, glob,sys
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import optimize
import pylab

#path = '/gpfs1/cusp/andyc/Night/'
path ='C:/software/PotPlayer 1.5.40688/Capture/Night/'
imlist = glob.glob( os.path.join(path, '*.png') )
dt = 1
L1 = [[166,212],[173,225]]
L2 = [[307,218],[314,231]]
car = [[210,196],[265,239]]
L1_var = []
L2_var = []
car_var = []
for i in range(len(imlist)-dt):
    print i
    im1 = np.array(Image.open(imlist[i+dt]).convert('L')).astype(np.float)
    im2 = np.array(Image.open(imlist[i]).convert('L')).astype(np.float)
    diff = im1-im2;
    L1_cut = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0]]
    L2_cut = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0]]
    car_cut =diff[car[0][1]:car[1][1],car[0][0]:car[1][0]]
    L1_var.append(L1_cut.var()/2)
    L2_var.append(L2_cut.var()/7.5)
    car_var.append(car_cut.var())
    
figure(1,figsize=[7.5,7.5]),
plot(range(len(L1_var)),L1_var,color = '#990000',lw=2)
plot(range(len(L1_var)),L2_var, color = '#006600',lw=2)
fill_between(range(len(L1_var)),car_var,facecolor = '#0099FF',edgecolor='#0000FF')
plt.grid(b=1,lw =2)
plt.xlabel('time [s]')
plt.ylabel('region variance [arb units]')
plot([119,119],[0,500],color='black',lw=2)
pylab.ylim([0,500])
pylab.xlim([110,163])

plt.text(145,405, 'vehicle region',color = '#0000FF',fontsize = 12)
plt.text(145,420, 'traffic light region 1',color = '#006600',fontsize = 12)
plt.text(145,435, 'traffic light region 2',color = '#990000',fontsize = 12)

ticks = np.arange(0, len(imlist), len(imlist)/23)
labels = range(ticks.size)
plt.xticks(ticks[16:], labels[16:])

