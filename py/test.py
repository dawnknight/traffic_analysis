import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image

path ='/home/andyc/image/Night_frame/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
nrow,ncol,nband = nd.imread(imlist[0]).shape
im = np.zeros([nrow,ncol])
im1 = np.zeros([nrow,ncol])
imr = np.zeros([nrow,ncol])

bg = np.zeros([nrow,ncol])

idx =0

if not os.path.isfile('/home/andyc/image/test/BG.jpg'):
    for i in range(len(imlist)):
        im1[:,:] = np.array(Image.open(imlist[i]).convert('L')).astype(float)
        im = im +im1
        print i
    bg = im/len(imlist)
    f = Image.fromarray(bg.astype(np.uint8))
    f.save('/home/andyc/image/test/BG.jpg')
else:
    bg = np.array(Image.open('/home/andyc/image/test/BG.jpg').convert('L')).astype(float)




for ii in range(len(imlist)):
    

    im1[:,:] = np.array(Image.open(imlist[ii]).convert('L')).astype(float)-bg

    im1_f = im1.flatten() 
    temp = [-30 if im1_f[i]<-30 else im1_f[i] for i in range(len(im1_f))]     
    temp = [45 if temp[i]>45 else temp[i] for i in range(len(im1_f))]
    temp =(np.array(temp)+30)*255/75
    tmp = np.array(temp).reshape(nrow,ncol)

    print ii 
    f = Image.fromarray(tmp.astype(np.uint8))
    f.save('/home/andyc/image/test/%.5d.jpg'%ii)

