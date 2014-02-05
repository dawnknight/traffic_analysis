# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:08:00 2014

@author: atc327
"""
import os, glob,sys,pylab,pickle
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import optimize

f = file('shift.pkl')
sft = pickle.load(f)

    
#path = '/gpfs1/cusp/andyc/Night/'
path ='C:/Users/atc327/Desktop/Traffic data/Day/resize/'
imlist = glob.glob( os.path.join(path, '*.png') )
dt = 1

L1 = [[343,414],[348,427]]
L2 = [[510,379],[515,394]]
car = [[415,353],[468,404]]
ped = [[341,562],[410,615]]
L1_var = []
L2_var = []
car_var = []
ped_var = []
im1 = np.zeros([750,900])
im2 = np.zeros([750,900])



for i in range(len(imlist)-dt):
      print i
      im1 = np.array(Image.open(imlist[i+dt]).convert('L')).astype(np.float)
      im2 = np.array(Image.open(imlist[i]).convert('L')).astype(np.float)
       
#        diff = im1-im2
#        L1_cut = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0]]
#        L2_cut = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0]]
#        car_cut =diff[car[0][1]:car[1][1],car[0][0]:car[1][0]]
#        ped_cut =diff[ped[0][1]:ped[1][1],ped[0][0]:ped[1][0]]
     
      
#      L1_cut = im1[L1[0][1]-sft[i+dt][0]:L1[1][1]-sft[i+dt][0],L1[0][0]-sft[i+dt][1]:L1[1][0]-sft[i+dt][1]]-\
#               im2[L1[0][1]-sft[i][0]:L1[1][1]-sft[i][0],L1[0][0]-sft[i][1]:L1[1][0]-sft[i][1]]
#      L2_cut = im1[L2[0][1]-sft[i+dt][0]:L2[1][1]-sft[i+dt][0],L2[0][0]-sft[i+dt][1]:L2[1][0]-sft[i+dt][1]]-\
#               im2[L2[0][1]-sft[i][0]:L2[1][1]-sft[i][0],L2[0][0]-sft[i][1]:L2[1][0]-sft[i][1]] 
#      car_cut = im1[car[0][1]-sft[i+dt][0]:car[1][1]-sft[i+dt][0],car[0][0]-sft[i+dt][1]:car[1][0]-sft[i+dt][1]]-\
#               im2[car[0][1]-sft[i][0]:car[1][1]-sft[i][0],car[0][0]-sft[i][1]:car[1][0]-sft[i][1]]
#      ped_cut = im1[ped[0][1]-sft[i+dt][0]:ped[1][1]-sft[i+dt][0],ped[0][0]-sft[i+dt][1]:ped[1][0]-sft[i+dt][1]]-\
#               im2[ped[0][1]-sft[i][0]:ped[1][1]-sft[i][0],ped[0][0]-sft[i][1]:ped[1][0]-sft[i][1]] 
      L1_cut = im1[L1[0][1]+sft[i+dt][0]:L1[1][1]+sft[i+dt][0],L1[0][0]+sft[i+dt][1]:L1[1][0]+sft[i+dt][1]]-\
               im2[L1[0][1]+sft[i][0]:L1[1][1]+sft[i][0],L1[0][0]+sft[i][1]:L1[1][0]+sft[i][1]]
      L2_cut = im1[L2[0][1]+sft[i+dt][0]:L2[1][1]+sft[i+dt][0],L2[0][0]+sft[i+dt][1]:L2[1][0]+sft[i+dt][1]]-\
               im2[L2[0][1]+sft[i][0]:L2[1][1]+sft[i][0],L2[0][0]+sft[i][1]:L2[1][0]+sft[i][1]] 
      car_cut = im1[car[0][1]+sft[i+dt][0]:car[1][1]+sft[i+dt][0],car[0][0]+sft[i+dt][1]:car[1][0]+sft[i+dt][1]]-\
               im2[car[0][1]+sft[i][0]:car[1][1]+sft[i][0],car[0][0]+sft[i][1]:car[1][0]+sft[i][1]]
      ped_cut = im1[ped[0][1]+sft[i+dt][0]:ped[1][1]+sft[i+dt][0],ped[0][0]+sft[i+dt][1]:ped[1][0]+sft[i+dt][1]]-\
               im2[ped[0][1]+sft[i][0]:ped[1][1]+sft[i][0],ped[0][0]+sft[i][1]:ped[1][0]+sft[i][1]]
      L1_var.append(L1_cut.var())
      L2_var.append(L2_cut.var())
      car_var.append(car_cut.var())
      ped_var.append(ped_cut.var()) 
              
figure(1,figsize=[7.5,7.5]),
plot(range(len(L1_var)),L1_var,color = '#990000',lw=2)
plot(range(len(L1_var)),L2_var, color = '#006600',lw=2)
#plt.plot(range(len(L1_var)),ped_var, color = 'black',lw=2)
fill_between(range(len(L1_var)),car_var,facecolor = '#0099FF',edgecolor='#0000FF')

plt.grid(b=1,lw =2)
plt.xlabel('time [s]')
plt.ylabel('region variance [arb units]')
#plot([119,119],[0,500],color='black',lw=2)
#pylab.ylim([0,500])
pylab.xlim([500,674])

#plt.text(145,405, 'vehicle region',color = '#0000FF',fontsize = 12)
#plt.text(145,420, 'traffic light region 1',color = '#006600',fontsize = 12)
#plt.text(145,435, 'traffic light region 2',color = '#990000',fontsize = 12)
#
#ticks = np.arange(0, len(imlist), len(imlist)/27)
#labels = range(ticks.size)
#plt.xticks(ticks[20:], labels[20:])

