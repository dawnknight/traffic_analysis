# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 21:16:19 2014

@author: atc327
"""


import os, glob,sys,pylab,pickle
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from multiprocessing import Pool


f = file('C:/Users/atc327/Documents/GitHub/traffic_analysis/shift.pkl')
sft = pickle.load(f)

    
#path = '/gpfs1/cusp/andyc/Night/'
path ='C:/Users/atc327/Desktop/Traffic data/Day/day/'
imlist = glob.glob( os.path.join(path, '*.jpg') )


H,W,O = np.array(Image.open(imlist[0])).shape
bord =30
im1 = np.zeros([H+2*bord,W+2*bord,O])
im2 = np.zeros([H+2*bord,W+2*bord,O])
diff = np.zeros([H+2*bord,W+2*bord,O])
L1 = [[1357,1115],[1399,1135]]
L2 = [[1242,1659],[1289,1678]]
car = [[1194,1381],[1343,1522]]

L1_cut_R = np.zeros([42,20])
L2_cut_R = np.zeros([47,19])
car_cut_R = np.zeros([149,141])

L1_cut_G = np.zeros([42,20])
L2_cut_G = np.zeros([47,19])
car_cut_G = np.zeros([149,141])

L1_cut_B = np.zeros([42,20])
L2_cut_B = np.zeros([47,19])
car_cut_B = np.zeros([149,141])

def traffic_process(idx):

      im1[30:2487,30:2967] = np.array(Image.open(imlist[idx+1])).astype(np.float)
      im2[30:2487,30:2967] = np.array(Image.open(imlist[idx])).astype(np.float)    
      im1 = np.roll(np.roll(im1,sft[idx+1][0],axis=0),sft[idx+1][1],axis=1)
      im2 = np.roll(np.roll(im2,sft[idx][0],axis=0),sft[idx][1],axis=1)             
      diff = im1-im2;   
#      return idx 
      L1_cut_R = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]
      L2_cut_R = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]
      car_cut_R =diff[car[0][1]:car[1][1],car[0][0]:car[1][0],0]
      
#======================================================================
      L1_cut_G = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]
      L2_cut_G = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]
      car_cut_G =diff[car[0][1]:car[1][1],car[0][0]:car[1][0],1]
     
#========================================================================
      L1_cut_B = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]
      L2_cut_B = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]
      car_cut_B =diff[car[0][1]:car[1][1],car[0][0]:car[1][0],2]

      return[L1_cut_R.var(),L1_cut_G.var(),L1_cut_B.var(),\
             L2_cut_R.var(),L2_cut_G.var(),L2_cut_B.var(),\
             car_cut_R.var(),car_cut_G.var(),car_cut_B.var(),\
             L1_cut_R.mean(),L1_cut_G.mean(),L1_cut_B.mean(),\
             L2_cut_R.mean(),L2_cut_G.mean(),L2_cut_B.mean(),\
             car_cut_R.mean(),car_cut_G.mean(),car_cut_B.mean()]
       


if __name__=='__main__':  



    result =[]
    pool = Pool(processes=16)              
    result.append(pool.map(traffic_process,range(5)))#len(imlist)))) 
    
#diff = [np.abs(x-y) for x,y in zip(L1_mean_R,L1_mean_G)]
#L1_var_R_mdfy = [L1_var_R[i] if diff[i]>4 else 0 for i in range(len(L1_var_R))]
#L2_var_R_mdfy = [L2_var_R[i] if diff[i]>4 else 0 for i in range(len(L1_var_R))]    
    
      
#    idx = [i if (L1_mean_R[i]*L1_mean_G[i])<0 else -99  for i in range(len(L1_var_R))]
#    idx[:] = (value for value in idx if value != -99)
#    
#    v =  [(L1_mean_R[i]*L1_mean_G[i]) if (L1_mean_R[i]*L1_mean_G[i])<0 else -99  for i in range(len(L1_var_R))]
#    v[:] = (value for value in v  if value != -99)
#    sub_idx = [i for i,x in enumerate(v) if x >-1]
#     
#    
#            
#    figure(1,figsize=[7.5,7.5]),
#    plot(range(len(L1_var_R)),L1_var_R_mdfy,color = '#990000',lw=2)
#    plot(range(len(L1_var_R)),L2_var_R_mdfy, color = '#006600',lw=2)
#    fill_between(range(len(L1_var_R)),car_var_R,facecolor = '#0099FF',edgecolor='#0000FF')
#    
#    plt.grid(b=1,lw =2)
#    plt.xlabel('time [s]')
#    plt.ylabel('region variance [arb units]')
#    pylab.xlim([500,674])
#    title('Red')
#    
#    figure(2,figsize=[7.5,7.5]),
#    plot(range(len(L1_var_R)),L1_var_G,color = '#990000',lw=2)
#    plot(range(len(L1_var_R)),L2_var_G, color = '#006600',lw=2)
#    fill_between(range(len(L1_var_R)),car_var_G,facecolor = '#0099FF',edgecolor='#0000FF')
#    
#    plt.grid(b=1,lw =2)
#    plt.xlabel('time [s]')
#    plt.ylabel('region variance [arb units]')
#    pylab.xlim([500,674])
#    title('Green')
#    
#    figure(3,figsize=[7.5,7.5]),
#    plot(range(len(L1_var_R)),L1_var_B,color = '#990000',lw=2)
#    plot(range(len(L1_var_R)),L2_var_B, color = '#006600',lw=2)
#    fill_between(range(len(L1_var_R)),car_var_B,facecolor = '#0099FF',edgecolor='#0000FF')
#    
#    plt.grid(b=1,lw =2)
#    plt.xlabel('time [s]')
#    plt.ylabel('region variance [arb units]')
#    pylab.xlim([500,674])
#    title('Blue')


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

