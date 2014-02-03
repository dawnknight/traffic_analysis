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
# -- utilities
bord  = 20
nside = 401 # npix/side of a postage stamp
reg   = (0, 0, 400, 400) # (ul row, ul col, lr row, lr col)


# -- set the reference frame (registering off of the green image)
rpath = 'D:/dropbox/CUSP/resize'
rfile ='D:/dropbox/CUSP/resize/0537.png' 
im = np.array(Image.open(rfile).convert('L')).astype(np.float)
ref   = 1.0*im[reg[0]:reg[2],reg[1]:reg[3]]
ref  -= ref.mean()
ref  /= ref.std()
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
imlist = glob.glob( os.path.join(rpath, '*.png') )
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
       off  = [mind / nside - nside//2, mind % nside ]
       cc_sub_dic[i] = off

        # -- send sub-matrix back to parent
#   if multi:
#      conn.send([cc_sub_mat,cc_sub_dic])
#      conn.close()
#   else:
#      return cc_sub_mat, cc_sub_dic

f = file('shift.pkl','wb')
pickle.dump(cc_sub_dic,f)
f.close()

