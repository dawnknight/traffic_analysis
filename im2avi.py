# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:46:43 2014

@author: atc327
"""

import cv
import os, glob

path = 'C:/software/PotPlayer 1.5.40688/Capture/Night/'
Fname = 'output.avi'
video = cv.CreateVideoWriter(Fname,cv.CV_FOURCC('X','V','I','D') , 10,(2936,2456), 1)
for infile in glob.glob( os.path.join(path, '*.jpg') ):
    img = cv.LoadImage(infile).convert
    cv.WriteFrame(video, img)
del video

