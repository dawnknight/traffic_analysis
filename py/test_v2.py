import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image

path ='/home/andyc/image/Feb11/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
nrow,ncol,nband = nd.imread(imlist[0]).shape
im = np.zeros([nrow,ncol])
im1 = np.zeros([nrow,ncol])
imr = np.zeros([nrow,ncol])
ima = np.zeros([nrow,ncol])

trunc = 8628
bg = np.zeros([nrow,ncol])

lb =-30
ub = 45


for i in range(trunc):
    im1[:,:] = np.array(Image.open(imlist[i]).convert('L')).astype(float)
    im = im +im1
    print i 

bg[:,:] = im/trunc

for ii in range(len(imlist)):
    print ii
    if ii < ceil(trunc/2):
        im1[:,:] = np.array(Image.open(imlist[ii]).convert('L')).astype(float)-bg
    elif (ii >= ceil(trunc/2)) & (ii+floor(trunc/2) < len(imlist)):
        ima = np.array(Image.open(imlist[ii+int(floor(trunc/2))]).convert('L')).astype(float)/trunc        
        imr = np.array(Image.open(imlist[ii-int(floor(trunc/2))-1]).convert('L')).astype(float)/trunc
        bg  = bg+ima-imr 
        im1[:,:] = np.array(Image.open(imlist[ii]).convert('L')).astype(float)-bg 
    else:
        im1[:,:] = np.array(Image.open(imlist[ii]).convert('L')).astype(float)-bg

    im1_f = im1.flatten()
    temp = [lb if im1_f[i]<lb else im1_f[i] for i in range(len(im1_f))]
    temp = [ub if temp[i]>ub else temp[i] for i in range(len(im1_f))]
    temp =(np.array(temp)-lb)*255/(ub-lb)
    tmp = np.array(temp).reshape(nrow,ncol)
    f = Image.fromarray(tmp.astype(np.uint8))
    f.save('/home/andyc/image/test/%.5d.jpg'%ii) 
