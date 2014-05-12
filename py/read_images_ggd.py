import os, sys
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

path = "/home/andyc/image/Mar10"

filelist = np.sort(os.listdir(path))

ncol  = 1280
nrow  = 720
nband = 3

img = np.zeros([nrow,ncol,nband])
img[:,:,:] = nd.imread(os.path.join(path,filelist[-1]))


for ii in filelist[-10:]:
    print("reading file: {0}".format(ii))
    img[:,:,:] = nd.imread(os.path.join(path,ii))
    print("file {0} read...".format(ii))
