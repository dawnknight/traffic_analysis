import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import mode
from scipy.interpolate import interp1d

def Main():

    #RG_idx = pickle.load(open("RG_idx.pkl","rb"))
    #path ='/home/andyc/image/Feb11/'
    #RG_idx = [86,179,196,221]
    path ='/home/andyc/image/Mar10/'
    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
    H,W,O = nd.imread(imlist[0]).shape
    diff = np.zeros([H,W])
    im1  = np.zeros([H,W,O])
    im2  = np.zeros([H,W,O])
    Cimg = np.zeros([H,W*2,O])

    #for i in range(len(RG_idx)):
    #for i in range(1):
    for j in range(len(imlist)-1):
        #lb = RG_idx[i]-1
        #ub = RG_idx[i]+2
        #lb = 120
        #ub = 150  
        #for j in range(lb,ub):
            #print('in {0}/{1} idx set, image {2}/{3}'.format(i+1,len(RG_idx),j-lb+1,(ub-lb)))  
            print('image {0}'.format(j))
            im1 = nd.imread(imlist[j]).astype(np.float)
            im2 = nd.imread(imlist[j+1]).astype(np.float)  
            diff= ((im2[:,:,0]-im1[:,:,0])+300)/2.0 
            Cimg[:,0:W,:]   = im2
            Cimg[:,W:2*W,0] = diff
            Cimg[:,W:2*W,1] = diff
            Cimg[:,W:2*W,2] = diff
            result = Image.fromarray(Cimg.astype(numpy.uint8))
            result.save('/home/andyc/image/diff/{0}.jpg'.format(str(j).zfill(5)))
Main()
