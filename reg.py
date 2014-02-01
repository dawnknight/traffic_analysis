# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 18:35:53 2014

@author: greg
"""

import os,multiprocessing,glob
from PIL import Image
import pickle as pkl
import numpy as np
from scipy.signal import fftconvolve



def register(inpath=None, infile=None, outpath=None, outfile=None,\
               multi=False, cc_mat=None, cc_dic=None):

    """ Register a single image or all images between some start and
    end time.  Pixels shifts for registration are written to a pickled
    dictionary in which the keys are the filenames and the values are
    a list or [row,col] pixel shifts."""


    # -- utilities
    bord  = 20
    nside = 401 # npix/side of a postage stamp
    reg   = (0, 0, 400, 400) # (ul row, ul col, ll row, ll col)


    # -- set the reference frame (registering off of the green image)
    rpath = 'C:/Users/atc327/Desktop/Traffic data/Day/resize'
    rfile ='C:/Users/atc327/Desktop/Traffic data/Day/resize/0537.png' 
    im = np.array(Image.open(rfile).convert('L')).astype(np.float)
    ref   = 1.0*im[reg[0]:reg[2],reg[1]:reg[3]]
    ref  -= ref.mean()
    ref  /= ref.std()
    imlist = glob.glob( os.path.join(rpath, '*.png') )

    # -- initialize the correlation dictionary and matrix
    if cc_dic==None:
        cc_dic = {}
    if cc_mat==None:
        cc_mat = np.zeros([len(imlist),2*bord+1,2*bord+1])


    # -- loop through the files and calculate the (sub-)correlation matrix
    nproc  = 1 if not multi else multi
    nfiles = len(imlist)
    dind   = 1 if nproc==1 \
               else nfiles//nproc if nfiles%nproc==0 \
               else nfiles//(nproc-1)

    def reg_subset(conn,rpath,verbose=False):

        # -- initialize the postage stamp, correlation, & sub-matrix dictionary
        stm        = np.zeros([reg[2]-reg[0],reg[3]-reg[1]])
        conv_mat   = np.zeros([ref.shape[0],ref.shape[1]])
        cc_sub_mat = np.zeros([len(imlist),2*bord+1,2*bord+1])
        cc_sub_dic = {}

        # -- loop through files
        imlist = glob.glob( os.path.join(rpath, '*.png') )
        for i in range(len(imlist)):
            if verbose:
                print("DST_REGISTER: " + 
                      "registering file {0} of {1}".format(i+1,len(imlist)))

            # -- initialize failure flag
            fflag = False


            # -- shift and find correlation

            im = np.array(Image.open(imlist[i]).convert('L')).astype(np.float)
            stm[:,:] = 1.0*im[reg[0]:reg[2],reg[1]:reg[3]]
            mn_stm   = stm.mean()
            sd_stm   = stm.std()

            if (sd_stm/(mn_stm+1.0))<0.5:
                fflag = True
            else:
                stm -= mn_stm
                stm /= sd_stm

                conv_mat[:,:] = fftconvolve(ref, stm[::-1,::-1], 'same')
                cc_sub_mat[i] = conv_mat[nside//2-20:nside//2+21,
                                         nside//2-20:nside//2+21]

                # -- find the maximum correlation and add to the dictionary
                mind = conv_mat.argmax()
                off  = [mind / nside - nside//2, mind % nside - nside//2]

                if max(np.abs(off))>(bord-1):
                    fflag = True

            if fflag:
                print("DST_REGISTER: REGISTRATION FAILED!!!")
                print("DST_REGISTER: {0}". format(imlist[i]))


                off = [314,314]

            cc_sub_dic[f] = off

        # -- send sub-matrix back to parent
        if multi:
            conn.send([cc_sub_mat,cc_sub_dic])
            conn.close()
        else:
            return cc_sub_mat, cc_sub_dic


    # -- calculate the correlation matrix
    if multi:
        print("DST_REGISTER: running {0} processes...".format(nproc))

        # -- initialize the full correlation matrix and processes
        parents, childs, ps = [], [], []

        # -- initialize the pipes and processes, then start
        for ip in range(nproc):
            ptemp, ctemp = multiprocessing.Pipe()
            parents.append(ptemp)
            childs.append(ctemp)
            ps.append(multiprocessing.Process(target=reg_subset,
                                           args=(childs[ip],
                                                 paths[dind*ip:dind*(ip+1)], 
                                                 files[dind*ip:dind*(ip+1)]), 
                                              kwargs={'verbose':ip==0}))
            ps[ip].start()

        # -- collect the results, put into cc_mat, and rejoin
        for ip in range(nproc):
            cc_sub_mat, cc_sub_dic = parents[ip].recv()
            cc_mat[dind*ip:dind*(ip+1),:,:] = cc_sub_mat
            cc_dic.update(cc_sub_dic)
            ps[ip].join()
            print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
        cc_mat[:,:,:], cc_sub_dic = reg_subset(-314,paths,files,verbose=True)
        cc_dic.update(cc_sub_dic)


    return cc_dic



