import os, glob,sys
import scipy as sp
import numpy as np
import scipy.ndimage as nd


path = '/home/andyc/image/Apr 17/'
imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
tmp = nd.imread(imlist[12000]).mean(2)
f = np.zeros(tmp.shape)
R = [[688,198],[695,202]]
Y = [[686,208],[691,214]]
G = [[684,218],[693,224]]
MG = 0
MY = 0
MR = 0

Th = 50

for i in range(12000,len(imlist)):
    print i
    f[:,:] = nd.imread(imlist[i]).mean(2)
    MR = MR + f[R[0][1]:R[1][1],R[0][0]:R[1][0]].flatten().mean() 
    MY = MY + f[Y[0][1]:Y[1][1],Y[0][0]:Y[1][0]].flatten().mean()
    MG = MG + f[G[0][1]:G[1][1],G[0][0]:G[1][0]].flatten().mean()
     
MR = MR/(len(imlist)-12000)
MY = MY/(len(imlist)-12000)
MG = MG/(len(imlist)-12000)

RV = np.zeros((len(imlist)-12000))
YV = np.zeros((len(imlist)-12000))
GV = np.zeros((len(imlist)-12000))

RV2 = np.zeros((len(imlist)-12000))
YV2 = np.zeros((len(imlist)-12000))
GV2 = np.zeros((len(imlist)-12000))


for i in range(12000,len(imlist)):
    print i
    f[:,:] = nd.imread(imlist[i]).mean(2)
    if (f[R[0][1]:R[1][1],R[0][0]:R[1][0]].flatten().mean() - MR) > Th:
        RV[i-12000] = f[R[0][1]:R[1][1],R[0][0]:R[1][0]].flatten().mean()
    if (f[Y[0][1]:Y[1][1],Y[0][0]:Y[1][0]].flatten().mean() - MY) > Th:
        YV[i-12000] = f[Y[0][1]:Y[1][1],Y[0][0]:Y[1][0]].flatten().mean()
    if (f[G[0][1]:G[1][1],G[0][0]:G[1][0]].flatten().mean() - MG) > Th:
        GV[i-12000] = f[G[0][1]:G[1][1],G[0][0]:G[1][0]].flatten().mean()
 
    RV2[i-12000] = f[R[0][1]:R[1][1],R[0][0]:R[1][0]].flatten().mean()
    YV2[i-12000] = f[Y[0][1]:Y[1][1],Y[0][0]:Y[1][0]].flatten().mean()
    GV2[i-12000] = f[G[0][1]:G[1][1],G[0][0]:G[1][0]].flatten().mean()


figure(1)

plot(RV,color = 'red')
plot(YV,color = 'yellow')
plot(GV,color = 'Green')


figure(2)


plot(RV2,color = 'red')
plot(YV2,color = 'yellow')
plot(GV2,color = 'Green')
