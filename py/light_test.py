import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from scipy.stats import mode

def Rm(idx):
    label =[]
    for i in range(1,len(idx)):
        if (idx[i]-idx[i-1])<15:
            label.append(i-1)
    for i in range(0,len(label))[::-1]:
        idx=delete(idx,label[i])
    return idx


def Trans_Idx(mean_mtx): # mtx are both a N*3 arrays                                                      
    v_RG = np.r_[mean_mtx[::,1]>0] & np.r_[mean_mtx[::,0]<0] \
                 & np.r_[(mean_mtx[::,1]*mean_mtx[::,0])<-1]
    v_GR = np.r_[mean_mtx[::,1]<0] & np.r_[mean_mtx[::,0]>0] \
                 & np.r_[(mean_mtx[::,1]*mean_mtx[::,0])<-1]

    RG_idx = [i for i in range(len(mean_mtx)) if v_RG[i]==True]
    GR_idx = [i for i in range(len(mean_mtx)) if v_GR[i]==True]
    
    return Rm(RG_idx),Rm(GR_idx) 

def Trans_Var(var_mtx,trans_idx):
    tmp_R = np.zeros(len(var_mtx))
    tmp_G = np.zeros(len(var_mtx))
    tmp_B = np.zeros(len(var_mtx))
    for i in range(len(trans_idx)):
        tmp_R[trans_idx[i]] = var_mtx[::,0][trans_idx[i]]
        tmp_G[trans_idx[i]] = var_mtx[::,1][trans_idx[i]]
        tmp_B[trans_idx[i]] = var_mtx[::,2][trans_idx[i]]
    var_mtx[::,0] = tmp_R 
    var_mtx[::,1] = tmp_G
    var_mtx[::,2] = tmp_B

    return var_mtx     

def Local_Max(mtx):  # mtx is N*1 vector
    max_label =np.r_[True, mtx[1:] > mtx[:-1]] & np.r_[mtx[:-1] > mtx[1:], True]
    # filter the noise effect
    constrain = [True if mtx[max(0,i-5):min(i+5,len(mtx))].var()>1 else False for i in range(len(mtx))]      
    max_idx = [i for i in range(len(mtx)) if (max_label[i] & constrain[i]) ==True]

    return max_idx

def React_Time(mv_idx,RG_idx,mv_var,partial=0.2,ck_th=50): # mv_var is N*1 vector
    mv_idx = np.asarray(mv_idx)    
    mv_var = np.asarray(mv_var)
    RG_idx = np.asarray(RG_idx)
    
    period = np.zeros(len(mv_idx))
    for ii in range(len(mv_idx)):
        val = mv_var[mv_idx[ii]]*partial
        
        tmp = np.r_[mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]][::-1]<=val]

        if True in tmp:
            table =arange(max(0,mv_idx[ii]-ck_th),mv_idx[ii])[::-1]        
            period[ii] = [table[i] for i in range(len(tmp)) if tmp[i]==True][0]-RG_idx[ii]         
        else:
            # find the most frequent value
            num = ceil(mode(np.round(mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]],1))[0][0])
            tmp = np.r_[mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]]>num]
            table =arange(max(0,mv_idx[ii]-ck_th),mv_idx[ii])
            period[ii] = [table[i] for i in range(len(tmp)) if tmp[i]==True][0]-RG_idx[ii]
            print("At {0}: original {1} is too small".format(RG_idx[ii],val/partial))
            print("use most freq. number {0} as new threshold".format(num))
    return period

def Move_Idx(RG_idx,lcmax_idx):

    if type(RG_idx) != array:
       RG_idx = np.asarray(RG_idx)
    if type(lcmax_idx) != array:
       lcmax_idx =np.asarray(lcmax_idx)

    mv_idx = np.zeros(len(RG_idx))
    for ii in range(len(RG_idx)):
        tmp_idx = np.zeros(len(lcmax_idx))
        tmp_idx= np.r_[lcmax_idx>=RG_idx[ii]]
        if True in tmp_idx:
            mv_idx[ii] = [lcmax_idx[i] for i in range(len(lcmax_idx)) if tmp_idx[i]==True][0]
        else:
            print("At {0}".format(ii))

    return mv_idx  



def Main():
    L1_var = pickle.load(open("L1_var.pkl","rb"))
    L2_var = pickle.load(open("L2_var.pkl","rb"))
    car_var = pickle.load(open("car_var.pkl","rb"))
    L1_avg = pickle.load(open("L1_avg.pkl","rb"))
    L2_avg = pickle.load(open("L2_avg.pkl","rb"))

    L1_RG_idx,L1_GR_idx = Trans_Idx(np.asarray(L1_avg.values()))
    L2_RG_idx,L2_GR_idx = Trans_Idx(np.asarray(L2_avg.values()))
     
    L1_VAR_RG = Trans_Var(np.asarray(L1_var.values()),L1_RG_idx)
    L1_VAR_GR = Trans_Var(np.asarray(L1_var.values()),L1_GR_idx)
    L2_VAR_RG = Trans_Var(np.asarray(L2_var.values()),L2_RG_idx)
    L2_VAR_GR = Trans_Var(np.asarray(L2_var.values()),L2_GR_idx)


    car_VAR = nd.gaussian_filter(np.asarray(car_var.values()),3)
    lcmax_idx_R = Local_Max(car_VAR[::,0])    

    react_T = React_Time(Move_Idx(L1_RG_idx,lcmax_idx_R),L1_RG_idx,car_VAR[::,0])


     

    figure(1,figsize=[7.5,7.5]),
    plot(range(len(L1_var)),L1_VAR_RG[::,0],color = '#990000',lw=2)
    plot(range(len(L1_var)),L2_VAR_RG[::,0]/3, color = '#006600',lw=2)
    fill_between(range(len(L1_var)),car_VAR[::,0]/2,facecolor = '#0099FF',edgecolor='#0000FF')


    plt.grid(b=1,lw =2)
    plt.xlabel('frame No.')
    plt.ylabel('region variance [arb units]')
    title('Red')
    
    figure(2,figsize=[7.5,7.5]),
    plot(range(len(react_T)),react_T,color = '#990000',lw=2)
    plt.grid(b=1,lw =2)

    figure(3,figsize=[7.5,7.5]),
    plt.hist(react_T,bins=30)
    plt.grid(b=1,lw =2)   

Main()
