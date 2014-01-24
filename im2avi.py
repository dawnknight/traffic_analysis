# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:46:43 2014

@author: atc327
"""

import cv
import cv2
import os, glob,sys
from PIL import Image
path = 'C:/software/PotPlayer 1.5.40688/Capture/Night'
video = cv.CreateVideoWriter('output.avi',cv.CV_FOURCC('D', 'I', 'V', 'X') , 10,(2456,2936), 1)
for infile in glob.glob( os.path.join(path, '*.jpg') ):
    img = cv.LoadImage(infile)
    cv.WriteFrame(video, img)
del video

#import subprocess as sp
#from PIL import Image
#import os, glob,sys
#path = 'C:\software\PotPlayer 1.5.40688\Capture\Nigh
#
#
#
#for infile in glob.glob( os.path.join(path, '*.jpg') ):
#    im = Image.open(infile)
#
#pipe = sp.Popen([ FFMPEG_BIN,
#        '-y', # (optional) overwrite the output file if it already exists
#        '-f', 'rawvideo',
#        '-vcodec','rawvideo',
#        '-s', '420x360', # size of one frame
#        '-pix_fmt', 'rgb24',
#        '-r', '5', # frames per second
#        '-i', '-', # The imput comes from a pipe
#        '-an', # Tells FFMPEG not to expect any audio
#        '-vcodec', 'mpeg'",
#        'my_output_videofile.mp4' ],
#        stdin=sp.PIPE,stdout=sp.PIPE, stderr=sp.PIPE)