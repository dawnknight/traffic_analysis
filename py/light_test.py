import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt


def rm(idx,sub_idx):
    label =[]
    for i in range(1,len(sub_idx)):
        if (idx[sub_idx[i]]-idx[sub_idx[i-1]])<15:
            label.append(sub_idx[i-1])
    for i in range(1,len(label)):
        sub_idx.remove(label[i])
    return sub_idx

def select_value(mean_mtx,var_mtx): # mtx are both a N*3 arrays                                                                      \
                                                                                                                                      
    idx = [i if (mean_mtx[::,0][i]*mean_mtx[::,1][i])<0 else -99  for i in range(len(mean_mtx))]
    idx[:] = (value for value in idx if value != -99) #frame i and frame i+1 are in different sign                                   \
                                                                                                                                      
    v =  [(mean_mtx[::,0][i]*mean_mtx[::,1][i]) if (mean_mtx[::,1][i]*mean_mtx[::,0][i])<0 else -99  for i in range(len(mean_mtx))]
    v[:] = (value for value in v  if value != -99)
    sub_idx = [i for i,x in enumerate(v) if x <-1]    # index of idx                                                                 \
                                                                                                                                      
    sub_idx = rm(idx,sub_idx)

    tmp_R = np.zeros(len(var_mtx))
    tmp_G = np.zeros(len(var_mtx))

    for i in range(len(sub_idx)):
        tmp_R[idx[sub_idx[i]]] = var_mtx[::,0][idx[sub_idx[i]]]
        tmp_G[idx[sub_idx[i]]] = var_mtx[::,1][idx[sub_idx[i]]]

    var_mtx[::,0] = tmp_R
    var_mtx[::,1] = tmp_G

    return var_mtx
def Local_max(mtx):  # mtx is N*1 vector
    min_label =numpy.r_[True, mtx[1:] > mtx[:-1]] & numpy.r_[mtx[:-1] > mtx[1:], True]
    min_idx = [i if min_label[i]==True else -99  for i in range(len(mtx))]
    min_idx[:] = (value for value in min_idx if value != -99)
    return min_idx

def main():

    L1_var = pickle.load(open("L1_var.pkl","rb"))
    L2_var = pickle.load(open("L2_var.pkl","rb"))
    car_var = pickle.load(open("car_var.pkl","rb"))
    L1_avg = pickle.load(open("L1_avg.pkl","rb"))
    L2_avg = pickle.load(open("L2_avg.pkl","rb"))

    L1_VAR = select_value(np.asarray(L1_avg.values()),np.asarray(L1_var.values()))
    L2_VAR = select_value(np.asarray(L2_avg.values()),np.asarray(L2_var.values()))
    car_VAR = nd.gaussian_filter(np.asarray(car_var.values()),3)
    lcmax_idx_R = Local_max(car_VAR[::,0])
     
    

    figure(1,figsize=[7.5,7.5]),
    plot(range(len(L1_var)),L1_VAR[::,0],color = '#990000',lw=2)
    plot(range(len(L1_var)),L2_VAR[::,0]/3, color = '#006600',lw=2)
    fill_between(range(len(L1_var)),car_VAR[::,0]/2,facecolor = '#0099FF',edgecolor='#0000FF')


    plt.grid(b=1,lw =2)
    plt.xlabel('frame No.')
    plt.ylabel('region variance [arb units]')
    title('Red')
    
main()
