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

    path ='/home/andyc/image/Mar10/'
#    path ='/home/andyc/image/Feb11/'
#    path ='/home/andyc/image/Night_frame/'
    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))    
    
    nrow,ncol,nband = nd.imread(imlist[0]).shape

#    L1 = [[422,313],[431,338]] #Day 3/10
#    L2 = [[522,318],[529,338]] #Day 3/10
#    car =[[267,428],[459,577]] #Day 3/10

#    L1 = [[931,388],[1278,552]] #bike

#    L1 = [[515,302],[527,360]] #Day 3/10 angle3                                                                                     
#    L2 = [[718,359],[726,381]] #Day 3/10 angle3                                                                                     
#    car =[[0,436],[288,533]] #Day 3/10 angle3


#    L1 = [[949,189],[951,219]] #Day 3/10 angle5                                                                                          
#    L2 = [[1116,193],[1122,220]] #Day 3/10 angle5
#    car =[[94,342],[437,447]] #Day 3/10 angle5

#    L1 = [[723,310],[738,345]]    #Day 2/11
#    L2 = [[1185,267],[1197,303]]  #Day 2/11
#    car =[[932,167],[1042,330]]   #Day  2/11
#    L1 = [[815,394],[971,537]]   #Z crossing 2/11                       
#    L2 = [[764,546],[983,679]]   #car 2  2/11


#    L1 = [[673,852],[683,900]]   #Night 
#    L2 = [[1238,874],[1247,919]] #Night 
#    car =[[962,689],[1048,876]]  #Night


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


    def Tra_Sub_Ana(conn,sub_imlist,sub_imlist2,nstart,ip,nrow,ncol,nband):
        

        im1 = np.zeros([nrow,ncol,nband])
        im2 = np.zeros([nrow,ncol,nband])
        diff = np.zeros([nrow,ncol,nband])       
        diff_env = np.zeros([nrow,ncol,nband])
 
        L1_cut_R = np.zeros([L1[1][1]-L1[0][1],L1[1][0]-L1[0][0]])
        L2_cut_R = np.zeros([L2[1][1]-L2[0][1],L2[1][0]-L2[0][0]])
        car_cut_R = np.zeros([car[1][1]-car[0][1],car[1][0]-car[0][0]])
        env_sub_R = np.zeros([nrow,ncol])        

        L1_cut_G = np.zeros([L1[1][1]-L1[0][1],L1[1][0]-L1[0][0]])
        L2_cut_G = np.zeros([L2[1][1]-L2[0][1],L2[1][0]-L2[0][0]])
        car_cut_G = np.zeros([car[1][1]-car[0][1],car[1][0]-car[0][0]])
        env_sub_G =np.zeros([nrow,ncol])
        
        L1_cut_B = np.zeros([L1[1][1]-L1[0][1],L1[1][0]-L1[0][0]])
        L2_cut_B = np.zeros([L2[1][1]-L2[0][1],L2[1][0]-L2[0][0]])
        car_cut_B = np.zeros([car[1][1]-car[0][1],car[1][0]-car[0][0]])
        env_sub_B =np.zeros([nrow,ncol])

        L1_sub_var  ={}
        L2_sub_var  ={}
        car_sub_var  ={}
        env_sub_var  ={}

        L1_sub_avg  ={}
        L2_sub_avg  ={}
    

        for i in range(len(zip(sub_imlist,sub_imlist2))) : 
            
            print("Processor {0} analyze image {1}/{2}".format(ip,i,len(sub_imlist)))
            print("file name 1: {0}",sub_imlist2[i])
            print("file name 2: {0}",sub_imlist[i])
            
            im1[:,:,:] = nd.imread(sub_imlist2[i]).astype(np.float)
            im2[:,:,:] = nd.imread(sub_imlist[i]).astype(np.float)    
            
            diff[:,:,:] = im1-im2

            L1_cut_R[:,:]  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]
            L2_cut_R[:,:]  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]
            car_cut_R[:,:] = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],0]
            env_sub_R[:,:] = diff[:,:,0]

#======================================================================

            L1_cut_G[:,:]  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]
            L2_cut_G[:,:]  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]
            car_cut_G[:,:] = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],1]
            env_sub_G[:,:] = diff[:,:,1]

#========================================================================
            L1_cut_B[:,:]  = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]
            L2_cut_B[:,:]  = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]
            car_cut_B[:,:] = diff[car[0][1]:car[1][1],car[0][0]:car[1][0],2]
            env_sub_B[:,:] = diff[:,:,2]

            env_sub_R_var = Env_Var(L1_cut_R,L2_cut_R,car_cut_R,env_sub_R)
            env_sub_G_var = Env_Var(L1_cut_G,L2_cut_G,car_cut_G,env_sub_G)
            env_sub_B_var = Env_Var(L1_cut_B,L2_cut_B,car_cut_B,env_sub_B)

            L1_sub_var[nstart+i]  = [L1_cut_R.var(),L1_cut_G.var(),L1_cut_B.var()]    
            L2_sub_var[nstart+i]  = [L2_cut_R.var(),L2_cut_G.var(),L2_cut_B.var()]    
            car_sub_var[nstart+i]  = [car_cut_R.var(),car_cut_G.var(),car_cut_B.var()]
            env_sub_var[nstart+i]  = [env_sub_R_var,env_sub_G_var,env_sub_B_var]         

            L1_sub_avg[nstart+i]  = [L1_cut_R.mean(),L1_cut_G.mean(),L1_cut_B.mean()]    
            L2_sub_avg[nstart+i]  = [L2_cut_R.mean(),L2_cut_G.mean(),L2_cut_B.mean()]   


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
                                              ip*dind,ip,nrow,ncol,nband)))
                              
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




def main():
 
    L1_var,L2_var,car_var,env_var,L1_avg,L2_avg = Tra_Ana(16) 

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

#    pickle.dump(C1,open("./Feb11/count1.pkl","wb"),True) 
#    pickle.dump(C2,open("./Feb11/count2.pkl","wb"),True) 

    pickle.dump(L1_var,open("./Mar10/angle1/bike_var.pkl","wb"),True)                                                                 
#    pickle.dump(L2_var,open("./Mar10/angle3/L2_var.pkl","wb"),True)                                                                 
#    pickle.dump(car_var,open("./Mar10/angle3/car_var.pkl","wb"),True)
#    pickle.dump(env_var,open("./Mar10/angle3/env_var.pkl","wb"),True)
    pickle.dump(L1_avg,open("./Mar10/angle1/bike_avg.pkl","wb"),True)
#    pickle.dump(L2_avg,open("./Mar10/angle3/L2_avg.pkl","wb"),True)
    


tic = time.clock()
main()
toc = time.clock()
print("Total processing time is {0}".format(toc-tic))
