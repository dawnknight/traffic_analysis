# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:19:54 2014

@author: atc327
"""

import cv2

video = cv2.VideoCapture('C:/Users/atc327/Desktop/Traffic data/Night1/Night.avi')
if video.isOpened():
    rval,frame = video.read()
else:
    rval = False
idx = 0
while rval:
    print idx
    rval,frame = video.read()
    name = 'C:/Users/atc327/Desktop/Traffic data/Night1/frame/'+str(idx).zfill(3)+'.jpg'
    cv2.imwrite(name,frame)
    idx = idx+1 
video.release()