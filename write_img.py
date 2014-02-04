# -*- coding: utf-8 -*-
"""
Created on Mon Feb 03 15:34:23 2014

@author: atc327
"""

import sys,pickle
import numpy as np
from PIL import Image

f = file('sft.pkl')
sft = pickle.load(f)
im = np.zeros([747,892,3])
path ='C:/Users/atc327/Desktop/Traffic data/Day/resize/'
imlist = glob.glob( os.path.join(path, '*.png') )
My = -3
Mx = 8
for i in range(len(imlist)):
    im1 = np.array(Image.open(imlist[i])).astype(np.float)
    y,x = sft[i]
          
    im = im1[0+(y-My):746+(y-My),8+(x-Mx):899+(x-Mx),0:]
    im = Image.fromarray(im.astype(np.uint8))   
    im.save('C:/Users/atc327/Desktop/Traffic data/Day/reg/out(%.3d).png'%i)