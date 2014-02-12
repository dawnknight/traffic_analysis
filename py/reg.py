# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 20:23:41 2014

@author: Greg, Andy
"""

import os,multiprocessing,glob
from PIL import Image
import pickle as pkl
import numpy as np
from scipy.signal import fftconvolve
import pickle

multi=False
multi_ref = True

# -- utilities
bord  = 20
nside = 121.0 # npix/side of a postage stamp
reg   = (80, 80, 201, 201) # (ul row(Y), ul col(X), lr row(Y), lr col(X))
reg2 = (25,680,146,801)
reg3 = (500,45,621,166)
reg4 = (350,770,471,891)



rpath = 'C:/Users/atc327/Desktop/Traffic data/Day/resize/'
rfile = 'C:/Users/atc327/Desktop/Traffic data/Day/resize/0537.png' 

im = np.array(Image.open(rfile).convert('L')).astype(np.float)

ref   = 1.0*im[reg[0]:reg[2],reg[1]:reg[3]]
ref  -= ref.mean()
ref  /= ref.std()


if multi_ref:
    ref2   = 1.0*im[reg2[0]:reg2[2],reg2[1]:reg2[3]]
    ref2  -= ref2.mean()
    ref2  /= ref2.std()
    
    ref3   = 1.0*im[reg3[0]:reg3[2],reg3[1]:reg3[3]]
    ref3  -= ref3.mean()
    ref3  /= ref3.std()
    
    ref4   = 1.0*im[reg4[0]:reg4[2],reg4[1]:reg4[3]]
    ref4  -= ref4.mean()
    ref4  /= ref4.std()
    
    imlist = glob.glob( os.path.join(rpath, '*.png') )



cc_dic = {}
cc_mat = np.zeros([len(imlist),2*bord+1,2*bord+1])


    # -- loop through the files and calculate the (sub-)correlation matrix
nproc  = 1 if not multi else multi
nfiles = len(imlist)
dind   = 1 if nproc==1 \
           else nfiles//nproc if nfiles%nproc==0 \
           else nfiles//(nproc-1)

#    def reg_subset(conn,rpath,verbose=False):

        # -- initialize the postage stamp, correlation, & sub-matrix dictionary
stm        = np.zeros([reg[2]-reg[0],reg[3]-reg[1]])
conv_mat   = np.zeros([ref.shape[0],ref.shape[1]])
cc_sub_mat = np.zeros([len(imlist),2*bord+1,2*bord+1])
cc_sub_dic = {}


#imlist = glob.glob( os.path.join(rpath, '*.png') )
# -- loop through files
for i in range(len(imlist)):

       print("DST_REGISTER: " + "registering file {0} of {1}".format(i+1,len(imlist)))

       # -- shift and find correlation

       im = np.array(Image.open(imlist[i]).convert('L')).astype(np.float)
       stm[:,:] = 1.0*im[reg[0]:reg[2],reg[1]:reg[3]]
       mn_stm   = stm.mean()
       sd_stm   = stm.std()

       stm -= mn_stm
       stm /= sd_stm
       conv_mat[:,:] = fftconvolve(ref, stm[::-1,::-1], 'same')
       cc_sub_mat[i] = conv_mat[nside//2-20:nside//2+21, nside//2-20:nside//2+21]
       # -- find the maximum correlation and add to the dictionary
       mind = conv_mat.argmax()
       off  = [int(round(mind / nside) - nside//2), int(mind % nside- nside//2) ]  
       print off
       if multi_ref:
           
           stm[:,:] = 1.0*im[reg2[0]:reg2[2],reg2[1]:reg2[3]]
           mn_stm   = stm.mean()
           sd_stm   = stm.std()
    
           stm -= mn_stm
           stm /= sd_stm
           conv_mat[:,:] = fftconvolve(ref2, stm[::-1,::-1], 'same')
           cc_sub_mat[i] = conv_mat[nside//2-20:nside//2+21, nside//2-20:nside//2+21]
           # -- find the maximum correlation and add to the dictionary
           mind = conv_mat.argmax()
           off2  = [int(round(mind / nside) - nside//2), int(mind % nside- nside//2) ]           
           print off2 
           
           stm[:,:] = 1.0*im[reg3[0]:reg3[2],reg3[1]:reg3[3]]
           mn_stm   = stm.mean()
           sd_stm   = stm.std()
    
           stm -= mn_stm
           stm /= sd_stm
           conv_mat[:,:] = fftconvolve(ref3, stm[::-1,::-1], 'same')
           cc_sub_mat[i] = conv_mat[nside//2-20:nside//2+21, nside//2-20:nside//2+21]
           # -- find the maximum correlation and add to the dictionary
           mind = conv_mat.argmax()
           off3  = [int(round(mind / nside) - nside//2), int(mind % nside- nside//2) ]        
           print off3
           
           stm[:,:] = 1.0*im[reg4[0]:reg4[2],reg4[1]:reg4[3]]
           mn_stm   = stm.mean()
           sd_stm   = stm.std()
    
           stm -= mn_stm
           stm /= sd_stm
           conv_mat[:,:] = fftconvolve(ref4, stm[::-1,::-1], 'same')
           cc_sub_mat[i] = conv_mat[nside//2-20:nside//2+21, nside//2-20:nside//2+21]
           # -- find the maximum correlation and add to the dictionary
           mind = conv_mat.argmax()
           off4  = [int(round(mind / nside) - nside//2), int(mind % nside- nside//2) ] 
           print off4
       if multi_ref:
           off = [int(round((w+x+y+z)/4.0)) for w,x,y,z in zip(off,off2,off3,off4)]
           print off
       cc_sub_dic[i] = off

        # -- send sub-matrix back to parent
#   if multi:
#      conn.send([cc_sub_mat,cc_sub_dic])
#      conn.close()
#   else:
#      return cc_sub_mat, cc_sub_dic

f = file('shift_4ref.pkl','wb')
pickle.dump(cc_sub_dic,f)
f.close()

