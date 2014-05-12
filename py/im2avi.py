# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:46:43 2014

@author: atc327
"""

import cv
import os, glob

path = '/home/andyc/image/Apr 17/'
Fname = 'Timesquare.avi'
video = cv.CreateVideoWriter(Fname,cv.CV_FOURCC('X','V','I','D') , 30,(1280,720), 1)
#video = cv.CreateVideoWriter(Fname,cv.CV_FOURCC('H','F','Y','U') , 15,(290,220), 1)
imlist =  sorted(glob.glob( os.path.join(path, '*.jpg')) )
'''
for infile in sorted(glob.glob( os.path.join(path, '*.png')):
    print infile
    img = cv.LoadImage(infile)
    cv.WriteFrame(video, img)
del video
'''
for i in range(10000,10900):
    print i 
    img = cv.LoadImage(imlist[i])
    cv.WriteFrame(video, img) 
del video 
