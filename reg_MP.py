# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 18:47:14 2014

@author: atc327
"""

import os,multiprocessing,glob
from PIL import Image
import pickle as pkl
import numpy as np
from scipy.signal import fftconvolve
import pickle

multi=16
multi_ref = False
    
# -- utilities
bord  = 20
nside = 351.0 # npix/side of a postage stamp
reg   = (200, 300, 551, 651) # (ul row(Y), ul col(X), lr row(Y), lr col(X))
reg2 = (100,2350,451,2651)
reg3 = (1750,150,2101,501)
reg4 = (1100,2500,1451,2851)

rpath = 'C:/Users/atc327/Desktop/Traffic data/Day/day/'
rfile = 'C:/Users/atc327/Desktop/Traffic data/Day/day/0537.jpg' 
    
im = np.zeros([2457,2937,3])
    
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
        
imlist = glob.glob( os.path.join(rpath, '*.jpg') )
        
cc_dic = {}
cc_mat = np.zeros([len(imlist),2*bord+1,2*bord+1])



##############################################################################
def reg_subset(conn,imlist,nstart):

    stm        = np.zeros([reg[2]-reg[0],reg[3]-reg[1]])
    conv_mat   = np.zeros([ref.shape[0],ref.shape[1]])
    cc_sub_mat = np.zeros([len(imlist),2*bord+1,2*bord+1])
    cc_sub_dic = {}
    
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
    
           if multi_ref:
               off = [int(round((w+x+y+z)/4.0)) for w,x,y,z in zip(off,off2,off3,off4)]
     
#           cc_sub_dic[imlist[i]] = off
           cc_sub_dic[nstart+i] = off
     # -- send sub-matrix back to parent
    if multi:
            conn.send([cc_sub_mat,cc_sub_dic])
            conn.close()
    else:
            return cc_sub_mat, cc_sub_dic
      
################################################################################

if __name__=='__main__':  
    
    # -- loop through the files and calculate the (sub-)correlation matrix
    nproc  = 1 if not multi else multi
    nfiles = len(imlist)
    dind   = 1 if nproc==1 \
            else nfiles//nproc if nfiles%nproc==0 \
            else nfiles//(nproc-1)
    
    
    # -- initialize the postage stamp, correlation, & sub-matrix dictionary 
    
    if multi:
            # -- initialize the full correlation matrix and processes
            parents, childs, ps = [], [],[]

            # -- initialize the pipes and processes, then start
            for ip in range(nproc):
                ptemp, ctemp = multiprocessing.Pipe()
                parents.append(ptemp)
                childs.append(ctemp)
                ps.append(multiprocessing.Process(target=reg_subset,args=(childs[ip],imlist[dind*ip:dind*(ip+1)],ip*dind)))
                ps[ip].start()
    
            # -- collect the results, put into cc_mat, and rejoin
            for ip in range(nproc):
                cc_sub_mat, cc_sub_dic = parents[ip].recv()
                cc_mat[dind*ip:dind*(ip+1),:,:] = cc_sub_mat
                
                cc_dic.update(cc_sub_dic)
                ps[ip].join()
                print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
            cc_mat[:,:,:], cc_sub_dic = reg_subset(-314,imlist)
            cc_dic.update(cc_sub_dic)

    f = file('shiftMP.pkl','wb')
    pickle.dump(cc_dic,f)
    f.close()
