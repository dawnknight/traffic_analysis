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


L1_var = []
L2_var = []
car_var = []
ped_var = []
bord =10
im1 = np.zeros([750+2*bord,900+2*bord])
im2 = np.zeros([750+2*bord,900+2*bord])
#L1 = [[343+bord,414+bord],[348+bord,427+bord]]
#L2 = [[510+bord,379+bord],[515+bord,394+bord]]
#car = [[415+bord,353+bord],[468+bord,404+bord]]
#ped = [[341+bord,562+bord],[410+bord,615+bord]]
L1 = [[350,422],[357,436]]
L2 = [[517,388],[524,403]]
car = [[409,373],[474,417]]

for i in range(len(imlist)-dt):
      print i
      im1[10:760,10:910] = np.array(Image.open(imlist[i+dt]).convert('L')).astype(np.float)
      im2[10:760,10:910] = np.array(Image.open(imlist[i]).convert('L')).astype(np.float)
        
      im1 = np.roll(np.roll(im1,sft[i+dt][0],axis=0),sft[i+dt][1],axis=1)
      im2 = np.roll(np.roll(im2,sft[i][0],axis=0),sft[i][1],axis=1)       
        
      L1_cut = im1[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0]]-\
               im2[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0]]
      L2_cut = im1[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0]]-\
               im2[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0]]
      car_cut =im1[car[0][1]:car[1][1],car[0][0]:car[1][0]]-\
               im2[car[0][1]:car[1][1],car[0][0]:car[1][0]]

      im = Image.fromarray(im2[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0]].astype(np.uint8))   
      im.save('C:/Users/atc327/Desktop/Traffic data/Day/CUT/%.3d).png'%i)

#      L1_cut = im1[L1[0][1]+sft[i+dt][0]:L1[1][1]+sft[i+dt][0],L1[0][0]+sft[i+dt][1]:L1[1][0]+sft[i+dt][1]]-\
#               im2[L1[0][1]+sft[i][0]:L1[1][1]+sft[i][0],L1[0][0]+sft[i][1]:L1[1][0]+sft[i][1]]
#      L2_cut = im1[L2[0][1]+sft[i+dt][0]:L2[1][1]+sft[i+dt][0],L2[0][0]+sft[i+dt][1]:L2[1][0]+sft[i+dt][1]]-\
#               im2[L2[0][1]+sft[i][0]:L2[1][1]+sft[i][0],L2[0][0]+sft[i][1]:L2[1][0]+sft[i][1]] 
#      car_cut = im1[car[0][1]+sft[i+dt][0]:car[1][1]+sft[i+dt][0],car[0][0]+sft[i+dt][1]:car[1][0]+sft[i+dt][1]]-\
#               im2[car[0][1]+sft[i][0]:car[1][1]+sft[i][0],car[0][0]+sft[i][1]:car[1][0]+sft[i][1]]
#      ped_cut = im1[ped[0][1]+sft[i+dt][0]:ped[1][1]+sft[i+dt][0],ped[0][0]+sft[i+dt][1]:ped[1][0]+sft[i+dt][1]]-\
#               im2[ped[0][1]+sft[i][0]:ped[1][1]+sft[i][0],ped[0][0]+sft[i][1]:ped[1][0]+sft[i][1]]
      L1_var.append(L1_cut.var())
      L2_var.append(L2_cut.var())
      car_var.append(car_cut.var())
#      ped_var.append(ped_cut.var()) 
              
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

