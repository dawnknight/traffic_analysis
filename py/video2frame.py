# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:19:54 2014

@author: atc327
"""

import cv2

#for i in range(11,27):
if 1:
    fname = "/home/andyc/Videos/0507/MVI_0753.MOV"

    #print("video{0}".format(i))
    video = cv2.VideoCapture(fname)
    if video.isOpened():
        rval,frame = video.read()
    else:
        rval = False
    idx = 0
    while rval:
        print idx
        rval,frame = video.read()
        #name = '/home/andyc/image/Apr15/'+repr(i)+'/'+str(idx).zfill(5)+'.jpg'
        name = '/home/andyc/image/0507/'+str(idx).zfill(5)+'.jpg'
        #name = '/home/andyc/image/IR/EVENING/'+str(idx).zfill(5)+'.jpg'

        if rval:
            cv2.imwrite(name,frame)
        idx = idx+1 
    video.release()
