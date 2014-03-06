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
def Pts_Cnt(mtx,ub,lb):
    c = sum([1 for i in range(len(mtx)) if (mtx[i]>ub or mtx[i]<lb)])    
    return max(c,1)




def Tra_Ana(multi):

    path ='/home/andyc/image/Feb11/'
#    path ='/home/andyc/image/Night_frame/'
    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))    
    
    H,W,O = nd.imread(imlist[0]).shape

#    L1 = [[723,310],[738,345]]    #Day
#    L2 = [[1185,267],[1197,303]]  #Day
    car =[[932,167],[1042,330]]   #Day
    L1 = [[815,394],[971,537]]   #Z crossing                                                
    L2 = [[764,546],[983,679]]   #car 2


#    L1 = [[673,852],[683,900]]   #Night 
#    L2 = [[1238,874],[1247,919]] #Night 
#    car =[[962,689],[1048,876]]  #Night


    ref1 = nd.imread(imlist[600]).astype(np.float)
    ref2 = nd.imread(imlist[599]).astype(np.float)
    drefp_R = (ref1-ref2)[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0].flatten()
    drefc_R = (ref1-ref2)[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0].flatten()
    drefp_G = (ref1-ref2)[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1].flatten()
    drefc_G = (ref1-ref2)[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1].flatten()
    drefp_B = (ref1-ref2)[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2].flatten()
    drefc_B = (ref1-ref2)[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2].flatten()
    pub_R = drefp_R.mean()+5*drefp_R.std()     #ped upper bound                                                                          
    plb_R = drefp_R.mean()-5*drefp_R.std()     #ped lower bound                                                                          
    cub_R = drefc_R.mean()+5*drefc_R.std()     #car2 upper bound                                                                         
    clb_R = drefc_R.mean()-5*drefc_R.std()     #car2 lower bound                                                                          
    pub_G = drefp_G.mean()+5*drefp_G.std()     #ped upper bound                                                                           
    plb_G = drefp_G.mean()-5*drefp_G.std()     #ped lower bound                                                                           
    cub_G = drefc_G.mean()+5*drefc_G.std()     #car2 upper bound                                                                          
    clb_G = drefc_G.mean()-5*drefc_G.std()     #car2 lower bound 
    pub_B = drefp_B.mean()+5*drefp_B.std()     #ped upper bound                                                                           
    plb_B = drefp_B.mean()-5*drefp_B.std()     #ped lower bound                                                                           
    cub_B = drefc_B.mean()+5*drefc_B.std()     #car2 upper bound                                                                          
    clb_B = drefc_B.mean()-5*drefc_B.std()     #car2 lower bound 

    TH  = [[pub_R,plb_R,cub_R,clb_R],[pub_G,plb_G,cub_G,clb_G],[pub_B,plb_B,cub_B,clb_B]]              #threshold   




    L1_var  ={}
    L2_var  ={}
    car_var ={}
    env_var ={}
    L1_avg  ={}
    L2_avg  ={}
    C1 = {} #number of the change pts in region 1 
    C2 = {} #number of the change pts in region 2


    nproc  = 1 if not multi else multi
    nfiles = len(imlist)
    dind   = 1 if nproc==1 \
            else nfiles//nproc if nfiles%nproc==0 \
            else nfiles//(nproc-1)


    def Tra_Sub_Ana(conn,sub_imlist,sub_imlist2,nstart,ip,TH):

   
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
    
        L1_Cnt ={}
        L2_Cnt ={}
        for i in range(len(zip(sub_imlist,sub_imlist2))) : 
            
            print("Processor {0} analyze image {1}/{2}".format(ip,i,len(sub_imlist)))
    
            im1 = nd.imread(sub_imlist2[i]).astype(np.float)
            im2 = nd.imread(sub_imlist[i]).astype(np.float)    
            
            diff = im1-im2

            L1_cut_R  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]
            L2_cut_R  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]
            car_cut_R = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],0]
            env_sub_R = diff[:,:,0]

            C1_sub_R = Pts_Cnt(L1_cut_R.flatten(),TH[0][0],TH[0][1])
            C2_sub_R = Pts_Cnt(L2_cut_R.flatten(),TH[0][2],TH[0][3])

#======================================================================

            L1_cut_G  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]
            L2_cut_G  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]
            car_cut_G = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],1]
            env_sub_G = diff[:,:,1]
           
            C1_sub_G = Pts_Cnt(L1_cut_G.flatten(),TH[1][0],TH[1][1])
            C2_sub_G = Pts_Cnt(L2_cut_G.flatten(),TH[1][2],TH[1][3])

#========================================================================
            L1_cut_B  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]
            L2_cut_B  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]
            car_cut_B = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],2]
            env_sub_B = diff[:,:,2]
            
            C1_sub_B = Pts_Cnt(L1_cut_B.flatten(),TH[2][0],TH[2][1])
            C2_sub_B = Pts_Cnt(L2_cut_B.flatten(),TH[2][2],TH[2][3])

            env_sub_R_var = Env_Var(L1_cut_R,L2_cut_R,car_cut_R,env_sub_R)
            env_sub_G_var = Env_Var(L1_cut_G,L2_cut_G,car_cut_G,env_sub_G)
            env_sub_B_var = Env_Var(L1_cut_B,L2_cut_B,car_cut_B,env_sub_B)

            L1_sub_var[nstart+i]  = [L1_cut_R.var(),L1_cut_G.var(),L1_cut_B.var()]    
            L2_sub_var[nstart+i]  = [L2_cut_R.var(),L2_cut_G.var(),L2_cut_B.var()]    
            car_sub_var[nstart+i]  = [car_cut_R.var(),car_cut_G.var(),car_cut_B.var()]
            env_sub_var[nstart+i]  = [env_sub_R_var,env_sub_G_var,env_sub_B_var]         

            L1_sub_avg[nstart+i]  = [L1_cut_R.mean(),L1_cut_G.mean(),L1_cut_B.mean()]    
            L2_sub_avg[nstart+i]  = [L2_cut_R.mean(),L2_cut_G.mean(),L2_cut_B.mean()]    

            L1_Cnt[nstart+i] = [C1_sub_R,C1_sub_G,C1_sub_B]
            L2_Cnt[nstart+i] = [C2_sub_R,C2_sub_G,C2_sub_B]


            del  L1_cut_R,L2_cut_R,car_cut_R,env_sub_R
            del  L1_cut_G,L2_cut_G,car_cut_G,env_sub_G
            del  L1_cut_B,L2_cut_B,car_cut_B,env_sub_B  
            del  im1,im2,diff
          
        if multi:
            conn.send([L1_sub_var,L2_sub_var,car_sub_var,\
                       env_sub_var,L1_sub_avg,L2_sub_avg,L1_Cnt,L2_Cnt])
            conn.close()
        else:
            return L1_sub_var,L2_sub_var,car_sub_var,env_sub_var,L1_sub_avg,L2_sub_avg,L1_Cnt,L2_Cnt    



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
                                              ip*dind,ip,TH)))
                              
            ps[ip].start()
    
        # -- collect the results, put into cc_mat, and rejoin
        for ip in range(nproc):
            L1_sub_var,L2_sub_var,car_sub_var,env_sub_var,L1_sub_avg,L2_sub_avg,L1_Cnt,L2_Cnt = parents[ip].recv()
                
            L1_var.update(L1_sub_var)
            L2_var.update(L2_sub_var)
            car_var.update(car_sub_var)
            env_var.update(env_sub_var)
            L1_avg.update(L1_sub_avg)
            L2_avg.update(L2_sub_avg)
            C1.update(L1_Cnt)
            C2.update(L2_Cnt)
    

            ps[ip].join()
            print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
        L1_sub_var,L2_sub_var,car_sub_var,env_sub_var,L1_sub_avg,L2_sub_avg,L1_Cnt,L2_Cnt = Tra_Sub_Ana(-314,imlist)
        L1_var.update(L1_sub_var)
        L2_var.update(L2_sub_var)
        car_var.update(car_sub_var)
        env_var.update(env_sub_var)
        L1_avg.update(L1_sub_avg)
        L2_avg.update(L2_sub_avg)
        C1.update(L1_Cnt)
        C2.update(L2_Cnt)

    return L1_var,L2_var,car_var,env_var,L1_avg,L2_avg,C1,C2  




def main():

    L1_var,L2_var,car_var,env_var,L1_avg,L2_avg,C1,C2 = Tra_Ana(24)
 
#    pickle.dump(L1_var,open("./Feb11/L1_var.pkl","wb"),True)
#    pickle.dump(L2_var,open("./Feb11/L2_var.pkl","wb"),True)
#    pickle.dump(L1_var,open("./Feb11/ped_var.pkl","wb"),True)   # Z crossing
#    pickle.dump(L2_var,open("./Feb11/car2_var.pkl","wb"),True)  # Car 2
#    pickle.dump(car_var,open("./Feb11/car_var.pkl","wb"),True)
#    pickle.dump(env_var,open("./Feb11/env_var.pkl","wb"),True)
#    pickle.dump(L1_avg,open("./Feb11/L1_avg.pkl","wb"),True)
#    pickle.dump(L2_avg,open("./Feb11/L2_avg.pkl","wb"),True)
#    pickle.dump(L1_avg,open("./Feb11/ped_avg.pkl","wb"),True)   # Z crossing 
#    pickle.dump(L2_avg,open("./Feb11/car2_avg.pkl","wb"),True)  # Car 2   

    pickle.dump(C1,open("./Feb11/count1.pkl","wb"),True) 
    pickle.dump(C2,open("./Feb11/count2.pkl","wb"),True) 



tic = time.clock()
main()
toc = time.clock()
print("Total processing time is {0}".format(toc-tic))
