import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.interpolate import interp1d

def Rm(idx,var):
    label =[]
    MAX = []
    MAX_idx = []
    reset = True
    if len(idx)==1:
       label.append(idx[0])
    else:
        for i in range(1,len(idx)):
            if reset:
                MAX = var[idx[i-1]]
                MAX_idx = idx[i-1]
                reset = False
            if (idx[i]-idx[i-1])<15:
                if not MAX:
                    if var[idx[i]]>var[idx[i-1]]:
                        MAX = var[idx[i]] 
                        MAX_idx = idx[i]
                    else:
                        MAX = var[idx[i-1]]
                        MAX_idx = idx[i-1]
                else:
                    if var[idx[i]]>MAX:
                        MAX = var[idx[i]]
                        MAX_idx = idx[i]
                if i == len(idx)-1:
                    label.append(MAX_idx)
            else:
                label.append(MAX_idx)
                MAX =[]
                MAX_idx = []
                reset = True  
                if i == len(idx)-1:
                    label.append(idx[i])
    return label

def Trans_Idx(mean_mtx,var): # mtx are both a N*3 arrays

    RG = (mean_mtx[::,1]*mean_mtx[::,0])
    GR = (mean_mtx[::,1]*mean_mtx[::,0])                                                  
    v_RG = np.r_[mean_mtx[::,1]>0] & np.r_[mean_mtx[::,0]<0] \
                                   & np.r_[RG<-1]
    v_GR = np.r_[mean_mtx[::,1]<0] & np.r_[mean_mtx[::,0]>0] \
                                   & np.r_[GR<-1]
    RG_idx = np.array([i for i in range(len(mean_mtx)) if v_RG[i]==True])
    GR_idx = np.array([i for i in range(len(mean_mtx)) if v_GR[i]==True])
    
    return Rm(RG_idx,var[::,0]),Rm(GR_idx,var[::,0]) 

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
    Nsf = np.zeros(len(mv_idx))
    period = np.zeros(len(mv_idx))
    for ii in range(len(mv_idx)):
        val = mv_var[mv_idx[ii]]*partial
        
        tmp = np.r_[mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]][::-1]<=val]
        tmpE = np.r_[mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]][::-1]==val]
        table =arange(max(0,mv_idx[ii]-ck_th),mv_idx[ii])[::-1]
        if True in tmpE: 
            # number of start moving frame 
            Nsf[ii] = [table[i] for i in range(len(tmp)) if tmp[i]==True][0]
            period[ii] = Nsf[ii] - RG_idx[ii]
        else :
            if True in tmp:               
                LB = [table[i] for i in range(len(tmp)) if tmp[i]==True][0]         
                RB = LB+1
                Nsf[ii] = interp1d([mv_var[LB],mv_var[RB]],[LB,RB])(val) 
                period[ii] = Nsf[ii] - RG_idx[ii]                        
            else:
                # find the most frequent value around the traffic light impulse
                num = ceil(mode(np.round(mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]],1))[0][0])
                tmp = np.r_[mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]][::-1]<=num]
                tmpe = np.r_[mv_var[max(0,mv_idx[ii]-ck_th):mv_idx[ii]]==num]
                if True in tmpe:
                    Nsf[ii] = [table[i] for i in range(len(tmp)) if tmp[i]==True][0]
                    period[ii] = Nsf[ii] - RG_idx[ii]                      
                else :
                    LB = [table[i] for i in range(len(tmp)) if tmp[i]==True][0]
                    RB = LB+1
                    Nsf[ii] = interp1d([mv_var[LB],mv_var[RB]],[LB,RB])(num)
                    period[ii] = Nsf[ii] - RG_idx[ii]
                print("At {0}: original {1} is too small".format(RG_idx[ii],val))
                print("use most freq. number {0} as new threshold".format(num))

    return period,Nsf

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

def Bg_Ana(mtx,sidx,eidx,cnt=0):
    VM = np.zeros(len(sidx))
    for i in arange(len(sidx)):
        s = min(sidx[i],eidx[i]) 
        e = min(max(sidx[i],eidx[i])+1,len(mtx))
        if s!=e :
            if type(cnt) != array:
                VM[i] = mtx[s:e].mean()
            else:
                VM[i] = cnt[s:e].max()
        else:
            if type(cnt) != array:
                VM[i] = mtx[s]
            else:
                VM[i] = cnt[s].max  
    return VM

def Main():
    
#    fps = 10

#    L1_var = pickle.load(open("./Night/L1_var.pkl","rb"))
#    L2_var = pickle.load(open("./Night/L2_var.pkl","rb"))
#    car_var = pickle.load(open("./Night/car_var.pkl","rb"))
#    env_var = pickle.load(open("./Night/env_var.pkl","rb"))
#    L1_avg = pickle.load(open("./Night/L1_avg.pkl","rb"))
#    L2_avg = pickle.load(open("./Night/L2_avg.pkl","rb"))
   
#    fps = 4
#    L1_var = pickle.load(open("./Feb11/L1_var.pkl","rb"))
#    L2_var = pickle.load(open("./Feb11/L2_var.pkl","rb"))
#    car_var = pickle.load(open("./Feb11/car_var.pkl","rb"))
#    env_var = pickle.load(open("./Feb11/ped_var.pkl","rb"))
#    L1_avg = pickle.load(open("./Feb11/L1_avg.pkl","rb"))
#    L2_avg = pickle.load(open("./Feb11/L2_avg.pkl","rb"))
#    C1 = pickle.load(open("./Feb11/count1.pkl","rb"))
#    C2 = pickle.load(open("./Feb11/count2.pkl","rb"))


    fps = 30
    L1_var = pickle.load(open("./Mar10/L1_var.pkl","rb"))
    L2_var = pickle.load(open("./Mar10/L2_var.pkl","rb"))
    car_var = pickle.load(open("./Mar10/car_var.pkl","rb"))
    env_var = pickle.load(open("./Mar10/env_var.pkl","rb"))
    L1_avg = pickle.load(open("./Mar10/L1_avg.pkl","rb"))
    L2_avg = pickle.load(open("./Mar10/L2_avg.pkl","rb"))


    L1_RG_idx,L1_GR_idx = Trans_Idx(np.asarray(L1_avg.values()),np.asarray(L1_var.values()))
    L2_RG_idx,L2_GR_idx = Trans_Idx(np.asarray(L2_avg.values()),np.asarray(L2_var.values()))
     
    L1_VAR_RG = Trans_Var(np.asarray(L1_var.values()),L1_RG_idx)
    L1_VAR_GR = Trans_Var(np.asarray(L1_var.values()),L1_GR_idx)
    L2_VAR_RG = Trans_Var(np.asarray(L2_var.values()),L2_RG_idx)
    L2_VAR_GR = Trans_Var(np.asarray(L2_var.values()),L2_GR_idx)
    
#    env_VAR = nd.gaussian_filter(np.asarray(env_var.values()),3)
    env_VAR = np.asarray(env_var.values())
    #C1_cnt = np.asarray(C1.values())
    


    car_VAR = nd.gaussian_filter(np.asarray(car_var.values()),3)
    lcmax_idx_R = Local_Max(car_VAR[::,0])    

    react_T,Nsf = React_Time(Move_Idx(L1_RG_idx,lcmax_idx_R),L1_RG_idx,car_VAR[::,0])

    # env variance mean
    #env_VM= Bg_Ana(env_VAR[::,0],L1_RG_idx,np.round(Nsf),C1_cnt[::,0])
    env_VM= Bg_Ana(env_VAR[::,0],L1_RG_idx,np.round(Nsf)) 

    figure(1,figsize=[7.5,7.5]),

    plot(range(len(L1_var)),L1_VAR_RG[::,0],color = '#990000',lw=2)
    plot(range(len(L1_var)),L2_VAR_RG[::,0]/3, color = '#006600',lw=2)
    fill_between(range(len(L1_var)),car_VAR[::,0]/2,facecolor = '#0099FF',edgecolor='#0000FF')

    plt.grid(b=1,lw =2)
    plt.xlabel('Frame No.')
    plt.ylabel('Region Variance [arb units]')
    title('Traffic Light Transitions (Red->Green)')

    figure(2,figsize=[7.5,7.5]),
    fill_between(range(len(L1_var)),env_VAR[::,0],facecolor = '#cccccc',edgecolor='#000000')
    
    plt.grid(b=1,lw =2)
    plt.xlabel('Frame No.')
    plt.ylabel('Background Variance [arb units]')
    title('Traffic Light Transitions (Red->Green)')
    
    figure(3,figsize=[7.5,7.5]),
    plot(range(len(react_T)),react_T/fps,color = '#990000',lw=2)
    plt.grid(b=1,lw =2)

    plt.grid(b=1,lw =2)
    plt.xlabel('frame No.')
    plt.ylabel('Time [s]')
    title('Driver Reaction Time')

    figure(4,figsize=[7.5,7.5]),
    plt.hist(react_T/fps,bins=20)

    plt.grid(b=1,lw =2)   
    plt.xlabel('Time [s]')
    plt.ylabel('Number of Driver')
    title('Driver Reaction Time')

    figure(10,figsize=[7.5,7.5]),
    plot(log(env_VM),react_T/fps,'ro')

    plt.grid(b=1,lw =2)
    plt.xlabel('Variance of background [arb units]')
    plt.ylabel('Time spend [s]')
    title('Relation between Driver Reaction Time and background variance')
#    title('Relation between Driver Reaction Time and ped variance')
    for i in range(len(env_VM)):
        plt.text(log(env_VM[i]),react_T[i]/fps,i)
Main()
