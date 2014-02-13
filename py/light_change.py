# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:42:44 2014

@author: atc327
"""



import os, glob,sys,pylab,pickle,multiprocessing
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

def TRA_ana(multi):



    path ='/home/andyc/image/Feb11/'
    imlist_in = glob.glob( os.path.join(path, '*.jpg') )
    imlist = []
    for i in range(len(imlist_in)):
        name = "fc2_save_2014-02-11-150054-"+str(i).zfill(4)+".jpg"
        imlist.append(name)
    H,W,O = nd.imread(imlist[0]).shape
    bord =30

    L1 = [[723,310],[738,345]]
    L2 = [[1185,267],[1197,303]]


    L1_var  ={}
    L2_var  ={}
    L1_avg  ={}
    L2_avg  ={}

    nproc  = 1 if not multi else multi
    nfiles = len(imlist)
    dind   = 1 if nproc==1 \
            else nfiles//nproc if nfiles%nproc==0 \
            else nfiles//(nproc-1)


    def TRA_sub_ana(conn,sub_imlist,sub_imlist2,nstart):
    
        im1 = np.zeros([H,W,O])
        im2 = np.zeros([H,W,O])
        diff = np.zeros([H,W,O])
      
        L1_cut_R = np.zeros([35,15])
        L2_cut_R = np.zeros([36,12])
     
        L1_cut_G = np.zeros([35,15])
        L2_cut_G = np.zeros([36,12])
     
        L1_cut_B = np.zeros([35,15])
        L2_cut_B = np.zeros([36,12])
     
        L1_sub_var  ={}
        L2_sub_var  ={}

        L1_sub_avg  ={}
        L2_sub_avg  ={}

       
        for i in range(len(zip(sub_imlist,sub_imlist2))) : 
            
            print("anaylysis image {0}, Total image is {1}".format(i,len(sub_imlist)))
    
      
            im1 = nd.imread(sub_imlist2[i]).astype(np.float)
            im2 = nd.imread(sub_imlist[i]).astype(np.float)    
            
            diff = im1-im2

            L1_cut_R = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0]
            L2_cut_R = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0]
#======================================================================
            L1_cut_G = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1]
            L2_cut_G = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1]
#========================================================================
            L1_cut_B = diff[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],2]
            L2_cut_B = diff[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],2]
         
                    
            L1_sub_var[nstart+i]  = [L1_cut_R.var(),L1_cut_G.var(),L1_cut_B.var()]    
            L2_sub_var[nstart+i]  = [L2_cut_R.var(),L2_cut_G.var(),L2_cut_B.var()]    

         
            L1_sub_avg[nstart+i]  = [L1_cut_R.mean(),L1_cut_G.mean(),L1_cut_B.mean()]    
            L2_sub_avg[nstart+i]  = [L2_cut_R.mean(),L2_cut_G.mean(),L2_cut_B.mean()]    

          
            del  L1_cut_R,L2_cut_R,L1_cut_G,L2_cut_G,L1_cut_B,L2_cut_B  
            del  im1,im2,diff
          
        if multi:
            conn.send([L1_sub_var,L2_sub_var,L1_sub_avg,L2_sub_avg])
            conn.close()
        else:
            return L1_sub_var,L2_sub_varL1_sub_avg,L2_sub_avg    



################################################################################  

    if multi:
        # -- initialize the full correlation matrix and processes
        parents, childs, ps = [], [],[]

        # -- initialize the pipes and processes, then start
        print 'start!!'   
        for ip in range(nproc):
            ptemp, ctemp = multiprocessing.Pipe()
            parents.append(ptemp)
            childs.append(ctemp)
            ps.append(multiprocessing.Process(target=TRA_sub_ana,\
                                              args=(childs[ip],\
                                              imlist[dind*ip:dind*(ip+1)],\
                                              imlist[dind*ip+1:dind*(ip+1)+1],\
                                              ip*dind)))
                              
            ps[ip].start()
    
        # -- collect the results, put into cc_mat, and rejoin
        for ip in range(nproc):
            L1_sub_var,L2_sub_var,L1_sub_avg,L2_sub_avg = parents[ip].recv()
                
            L1_var.update(L1_sub_var)
            L2_var.update(L2_sub_var)
            L1_avg.update(L1_sub_avg)
            L2_avg.update(L2_sub_avg)
                

            ps[ip].join()
            print("DST_REGISTER: process {0} rejoined.".format(ip))
    else:
        L1_sub_var,L2_sub_var,L1_sub_avg,L2_sub_avg = TRA_sub_ana(-314,imlist)
        L1_var.update(L1_sub_var)
        L2_var.update(L2_sub_var)
        L1_avg.update(L1_sub_avg)
        L2_avg.update(L2_sub_avg)

    return L1_var,L2_var,L1_avg,L2_avg  



L1_var,L2_var,L1_avg,L2_avg = TRA_ana(24)

pickle.dump(L1_var,open("L1_var.pkl","wb"),True)
pickle.dump(L2_var,open("L2_var.pkl","wb"),True)
pickle.dump(L2_avg,open("L2_avg.pkl","wb"),True)
pickle.dump(L1_avg,open("L1_avg.pkl","wb"),True)



#L1_VAR = np.asarray(L1_var.values())
#L2_VAR = np.asarray(L2_var.values())
#L1_AVG = np.asarray(L1_avg.values())
#L2_AVG = np.asarray(L2_avg.values())


    
#diff = [np.abs(x-y) for x,y in zip(L1_mean_R,L1_mean_G)]
#L1_var_R_mdfy = [L1_var_R[i] if diff[i]>4 else 0 for i in range(len(L1_var_R))]
#L2_var_R_mdfy = [L2_var_R[i] if diff[i]>4 else 0 for i in range(len(L1_var_R))]    
    
      
#idx = [i if (L1_AVG[::,0][i]*L1_AVG[::,1][i])<0 else -99  for i in range(len(L1_var))]
#idx[:] = (value for value in idx if value != -99)
    
#v =  [(L1_AVG[::,0][i]*L1_AVG[::,1][i]) if (L1_AVG[::,0][i]*L1_AVG[::,1][i])<0 else -99  for i in range(len(L1_var))]
#v[:] = (value for value in v  if value != -99)
#sub_idx = [i for i,x in enumerate(v) if x >-1]
     
    
#            
#    figure(1,figsize=[7.5,7.5]),
#    plot(range(len(L1_var_R)),L1_var_R,color = '#990000',lw=2)
#    plot(range(len(L1_var_R)),L2_var_R, color = '#006600',lw=2)
#    fill_between(range(len(L1_var_R)),car_var_R,facecolor = '#0099FF',edgecolor='#0000FF')
    
#    plt.grid(b=1,lw =2)
#    plt.xlabel('time [s]')
#    plt.ylabel('region variance [arb units]')
#    title('Red')
    
#    figure(2,figsize=[7.5,7.5]),
#    plot(range(len(L1_var_R)),L1_var_G,color = '#990000',lw=2)
#    plot(range(len(L1_var_R)),L2_var_G, color = '#006600',lw=2)
#    fill_between(range(len(L1_var_R)),car_var_G,facecolor = '#0099FF',edgecolor='#0000FF')
    
#    plt.grid(b=1,lw =2)
#    plt.xlabel('time [s]')
#    plt.ylabel('region variance [arb units]')
#    title('Green')
    
#    figure(3,figsize=[7.5,7.5]),
#    plot(range(len(L1_var_R)),L1_var_B,color = '#990000',lw=2)
#    plot(range(len(L1_var_R)),L2_var_B, color = '#006600',lw=2)
#    fill_between(range(len(L1_var_R)),car_var_B,facecolor = '#0099FF',edgecolor='#0000FF')
    
#    plt.grid(b=1,lw =2)
#    plt.xlabel('time [s]')
#    plt.ylabel('region variance [arb units]')
#    title('Blue')


#plot([119,119],[0,500],color='black',lw=2)
#pylab.ylim([0,500])
#

#plt.text(145,405, 'vehicle region',color = '#0000FF',fontsize = 12)
#plt.text(145,420, 'traffic light region 1',color = '#006600',fontsize = 12)
#plt.text(145,435, 'traffic light region 2',color = '#990000',fontsize = 12)
#
#ticks = np.arange(0, len(imlist), len(imlist)/27)
#labels = range(ticks.size)
#plt.xticks(ticks[20:], labels[20:])

