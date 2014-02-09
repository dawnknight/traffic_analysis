# -*- coding: utf-8 -*-
"""
Created on Fri Feb 07 19:41:40 2014

@author: atc327
"""

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

f = file('C:/Users/atc327/Documents/GitHub/traffic_analysis/shift.pkl')
sft = pickle.load(f)

    
#path = '/gpfs1/cusp/andyc/Night/'
path ='C:/Users/atc327/Desktop/Traffic data/Day/resize/'
imlist = glob.glob( os.path.join(path, '*.png') )
dt = 1


L1_var_R = []
L2_var_R = []
car_var_R = []
L1_var_G = []
L2_var_G = []
car_var_G = []
L1_var_B = []
L2_var_B = []
car_var_B = []

bord =10
im1 = np.zeros([750+2*bord,900+2*bord,3])
im2 = np.zeros([750+2*bord,900+2*bord,3])
L1 = [[350,422],[357,436]]
L2 = [[517,388],[524,403]]
car = [[409,373],[474,417]]

for i in range(len(imlist)-dt):
      print i
      im1[10:760,10:910] = np.array(Image.open(imlist[i+dt])).astype(np.float)
      im2[10:760,10:910] = np.array(Image.open(imlist[i])).astype(np.float)
        
      im1 = np.roll(np.roll(im1,sft[i+dt][0],axis=0),sft[i+dt][1],axis=1)
      im2 = np.roll(np.roll(im2,sft[i][0],axis=0),sft[i][1],axis=1)       
        
      L1_cut_R = im1[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]-\
                 im2[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]
      L2_cut_R = im1[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]-\
                 im2[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]
      car_cut_R =im1[car[0][1]:car[1][1],car[0][0]:car[1][0],0]-\
                 im2[car[0][1]:car[1][1],car[0][0]:car[1][0],0]

      L1_var_R.append(L1_cut_R.var())
      L2_var_R.append(L2_cut_R.var())
      car_var_R.append(car_cut_R.var())
#======================================================================
      L1_cut_G = im1[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]-\
                 im2[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]
      L2_cut_G = im1[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]-\
                 im2[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]
      car_cut_G =im1[car[0][1]:car[1][1],car[0][0]:car[1][0],1]-\
                 im2[car[0][1]:car[1][1],car[0][0]:car[1][0],1]

      L1_var_G.append(L1_cut_G.var())
      L2_var_G.append(L2_cut_G.var())
      car_var_G.append(car_cut_G.var())
#========================================================================
      L1_cut_B = im1[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]-\
                 im2[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]
      L2_cut_B = im1[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]-\
                 im2[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]
      car_cut_B =im1[car[0][1]:car[1][1],car[0][0]:car[1][0],2]-\
                 im2[car[0][1]:car[1][1],car[0][0]:car[1][0],2]

      L1_var_B.append(L1_cut_B.var())
      L2_var_B.append(L2_cut_B.var())
      car_var_B.append(car_cut_B.var())


              
figure(1,figsize=[7.5,7.5]),
plot(range(len(L1_var_R)),L1_var_R,color = '#990000',lw=2)
plot(range(len(L1_var_R)),L2_var_R, color = '#006600',lw=2)
fill_between(range(len(L1_var_R)),car_var_R,facecolor = '#0099FF',edgecolor='#0000FF')

plt.grid(b=1,lw =2)
plt.xlabel('time [s]')
plt.ylabel('region variance [arb units]')
pylab.xlim([500,674])
title('Red')

figure(2,figsize=[7.5,7.5]),
plot(range(len(L1_var_R)),L1_var_G,color = '#990000',lw=2)
plot(range(len(L1_var_R)),L2_var_G, color = '#006600',lw=2)
fill_between(range(len(L1_var_R)),car_var_G,facecolor = '#0099FF',edgecolor='#0000FF')

plt.grid(b=1,lw =2)
plt.xlabel('time [s]')
plt.ylabel('region variance [arb units]')
pylab.xlim([500,674])
title('Green')

figure(3,figsize=[7.5,7.5]),
plot(range(len(L1_var_R)),L1_var_B,color = '#990000',lw=2)
plot(range(len(L1_var_R)),L2_var_B, color = '#006600',lw=2)
fill_between(range(len(L1_var_R)),car_var_B,facecolor = '#0099FF',edgecolor='#0000FF')

plt.grid(b=1,lw =2)
plt.xlabel('time [s]')
plt.ylabel('region variance [arb units]')
pylab.xlim([500,674])
title('Blue')


#plot([119,119],[0,500],color='black',lw=2)
#pylab.ylim([0,500])
#

#plt.text(145,405, 'vehicle region',color = '#0000FF',fontsize = 12)
#plt.text(145,420, 'traffic light region 1',color = '#006600',fontsize = 12)
#plt.text(145,435, 'traffic light region 2',color = '#990000',fontsize = 12)
#
#ticks = np.arange(0, len(imlist), len(imlist)/27)
#labels = range(ticks.size)
#plt.xticks(ticks[20:], labels[20:])

