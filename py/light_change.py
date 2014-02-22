# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:42:44 2014

@author: atc327
"""

import os, glob,sys,pylab,pickle,multiprocessing,time
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def Env_Var(r1,r2,r3,mtx):
    H1,W1 = r1.shape
    H2,W2 = r2.shape
    H3,W3 = r3.shape
    H,W   = mtx.shape
    Avg = (mtx.mean()*H*W-r1.mean()*H1*W1-r2.mean()*H2*W2-r3.mean()*H3*W3)/(H*W-H1*W1-H2*W2-H3*W3)
    Var = ((sum(mtx**2)-sum(r1**2)-sum(r2**2)-sum(r3**2))\
            -2*Avg*(sum(mtx)-sum(r1)-sum(r2)-sum(r3))\
            +(H*W-H1*W1-H2*W2-H3*W3)*Avg**2)\
            /(H*W-H1*W1-H2*W2-H3*W3)
    return Var 


def Tra_Ana(multi):

#    path ='/home/andyc/image/Feb11/'
    path ='/home/andyc/image/Night_frame/'
    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))    
    
    H,W,O = nd.imread(imlist[0]).shape

#    L1 = [[723,310],[738,345]]
#    L2 = [[1185,267],[1197,303]]
#    car =[[932,167],[1042,330]]

    L1 = [[673,852],[683,900]]
    L2 = [[1238,874],[1247,919]]
    car =[[962,689],[1048,876]]


    L1_var  ={}
    L2_var  ={}
    car_var ={}
    env_var ={}
    L1_avg  ={}
    L2_avg  ={}

    nproc  = 1 if not multi else multi
    nfiles = len(imlist)
    dind   = 1 if nproc==1 \
            else nfiles//nproc if nfiles%nproc==0 \
            else nfiles//(nproc-1)


    def Tra_Sub_Ana(conn,sub_imlist,sub_imlist2,nstart,ip):

   
        im1 = np.zeros([H,W,O])
        im2 = np.zeros([H,W,O])
        diff = np.zeros([H,W,O])       
        diff_env = np.zeros([H,W,O])
 
        L1_cut_R = np.zeros([L1[1][1]-L1[0][1],L1[1][0]-L1[0][0]])
        L2_cut_R = np.zeros([L2[1][1]-L2[0][1],L2[1][0]-L2[0][0]])
        car_cut_R = np.zeros([car[1][1]-car[0][1],car[1][0]-car[0][0]])
        env_R = np.zeros([H,W])        

        L1_cut_G = np.zeros([L1[1][1]-L1[0][1],L1[1][0]-L1[0][0]])
        L2_cut_G = np.zeros([L2[1][1]-L2[0][1],L2[1][0]-L2[0][0]])
        car_cut_G = np.zeros([110,163])
        env_G =np.zeros([H,W])
        
        L1_cut_B = np.zeros([L1[1][1]-L1[0][1],L1[1][0]-L1[0][0]])
        L2_cut_B = np.zeros([L2[1][1]-L2[0][1],L2[1][0]-L2[0][0]])
        car_cut_B = np.zeros([car[1][1]-car[0][1],car[1][0]-car[0][0]])
        env_B =np.zeros([H,W])

        L1_sub_var  ={}
        L2_sub_var  ={}
        car_sub_var  ={}
        env_sub_var  ={}

        L1_sub_avg  ={}
        L2_sub_avg  ={}
    
       
        for i in range(len(zip(sub_imlist,sub_imlist2))) : 
            
            print("Processor {0} analyze image {1}/{2}".format(ip,i,len(sub_imlist)))
    
            im1 = nd.imread(sub_imlist2[i]).astype(np.float)
            im2 = nd.imread(sub_imlist[i]).astype(np.float)    
            
            diff = im1-im2

            L1_cut_R  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]
            L2_cut_R  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]
            car_cut_R = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],0]
            env_sub_R = diff[:,:,0]

#======================================================================

            L1_cut_G  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]
            L2_cut_G  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]
            car_cut_G = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],1]
            env_sub_G = diff[:,:,1]

#========================================================================
            L1_cut_B  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]
            L2_cut_B  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]
            car_cut_B = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],2]
            env_sub_B = diff[:,:,2]
            
            env_sub_R_var = Env_Var(L1_cut_R,L2_cut_R,car_cut_R,env_sub_R)
            env_sub_G_var = Env_Var(L1_cut_G,L2_cut_G,car_cut_G,env_sub_G)
            env_sub_B_var = Env_Var(L1_cut_B,L2_cut_B,car_cut_B,env_sub_B)

            L1_sub_var[nstart+i]  = [L1_cut_R.var(),L1_cut_G.var(),L1_cut_B.var()]    
            L2_sub_var[nstart+i]  = [L2_cut_R.var(),L2_cut_G.var(),L2_cut_B.var()]    
            car_sub_var[nstart+i]  = [car_cut_R.var(),car_cut_G.var(),car_cut_B.var()]
            env_sub_var[nstart+i]  = [env_sub_R_var,env_sub_G_var,env_sub_B_var]         

            L1_sub_avg[nstart+i]  = [L1_cut_R.mean(),L1_cut_G.mean(),L1_cut_B.mean()]    
            L2_sub_avg[nstart+i]  = [L2_cut_R.mean(),L2_cut_G.mean(),L2_cut_B.mean()]    

          
            del  L1_cut_R,L2_cut_R,car_cut_R,env_sub_R
            del  L1_cut_G,L2_cut_G,car_cut_G,env_sub_G
            del  L1_cut_B,L2_cut_B,car_cut_B,env_sub_B  
            del  im1,im2,diff
          
        if multi:
            conn.send([L1_sub_var,L2_sub_var,car_sub_var,\
                       env_sub_var,L1_sub_avg,L2_sub_avg])
            conn.close()
        else:
            return L1_sub_var,L2_sub_var,car_sub_var,env_sub_var,L1_sub_avg,L2_sub_avg    



################################################################################  

    if multi:
        # -- initialize the full correlation matrix and processes
        parents, childs, ps = [], [],[]

        # -- initialize the pipes and processes, then start

        for ip in range(nproc):
            ptemp, ctemp = multiprocessing.Pipe()
            parents.append(ptemp)
            childs.append(ctemp)
            ps.append(multiprocessing.Process(target=Tra_Sub_Ana,\
                                              args=(childs[ip],\
                                              imlist[dind*ip:dind*(ip+1)],\
                                              imlist[dind*ip+1:dind*(ip+1)+1],\
                                              ip*dind,ip)))
                              
            ps[ip].start()
    
        # -- collect the results, put into cc_mat, and rejoin
        for ip in range(nproc):
            L1_sub_var,L2_sub_var,car_sub_var,env_sub_var,L1_sub_avg,L2_sub_avg = parents[ip].recv()
                
            L1_var.update(L1_sub_var)
            L2_var.update(L2_sub_var)
            car_var.update(car_sub_var)
            env_var.update(env_sub_var)
            L1_avg.update(L1_sub_avg)
            L2_avg.update(L2_sub_avg)
                

            ps[ip].join()
            print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
        L1_sub_var,L2_sub_var,car_sub_var,env_sub_var,L1_sub_avg,L2_sub_avg = Tra_Sub_Ana(-314,imlist)
        L1_var.update(L1_sub_var)
        L2_var.update(L2_sub_var)
        car_var.update(car_sub_var)
        env_var.update(env_sub_var)
        L1_avg.update(L1_sub_avg)
        L2_avg.update(L2_sub_avg)

    return L1_var,L2_var,car_var,env_var,L1_avg,L2_avg  

def rm(idx,sub_idx):
    label =[]
    for i in range(1,len(sub_idx)):
        if (idx[sub_idx[i]]-idx[sub_idx[i-1]])<15:
            label.append(sub_idx[i-1])
    for i in range(1,len(label)):    
        sub_idx.remove(label[i])
    return sub_idx

def select_value(mean_mtx,var_mtx): # mtx are both a N*3 arrays                                                                                           
    idx = [i if (mean_mtx[::,0][i]*mean_mtx[::,1][i])<0 else -99  for i in range(len(mean_mtx))]
    idx[:] = (value for value in idx if value != -99) #frame i and frame i+1 are in different sign                                    
    v =  [(mean_mtx[::,0][i]*mean_mtx[::,1][i]) if (mean_mtx[::,1][i]*mean_mtx[::,0][i])<0 else -99  for i in range(len(mean_mtx))]
    v[:] = (value for value in v  if value != -99)
    sub_idx = [i for i,x in enumerate(v) if x <-1]    # index of idx                                                                         
    sub_idx = rm(idx,sub_idx)     

    tmp_R = np.zeros(len(var_mtx))
    tmp_G = np.zeros(len(var_mtx))

    for i in range(len(sub_idx)):
        tmp_R[idx[sub_idx[i]]] = var_mtx[::,0][idx[sub_idx[i]]]
        tmp_G[idx[sub_idx[i]]] = var_mtx[::,1][idx[sub_idx[i]]]

    var_mtx[::,0] = tmp_R
    var_mtx[::,1] = tmp_G

    return var_mtx


def main():

    L1_var,L2_var,car_var,env_var,L1_avg,L2_avg = Tra_Ana(16)

    pickle.dump(L1_var,open("./Night/L1_var.pkl","wb"),True)
    pickle.dump(L2_var,open("./Night/L2_var.pkl","wb"),True)
    pickle.dump(car_var,open("./Night/car_var.pkl","wb"),True)
    pickle.dump(env_var,open("./Night/env_var.pkl","wb"),True)
    pickle.dump(L2_avg,open("./Night/L2_avg.pkl","wb"),True)
    pickle.dump(L1_avg,open("./Night/L1_avg.pkl","wb"),True)


tic = time.clock()
main()
toc = time.clock()
print("Total processing time is {0}".format(toc-tic))
