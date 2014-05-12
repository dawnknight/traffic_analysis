import cv2,os, glob,sys,pylab,pickle,multiprocessing
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory
import scipy as sp
import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from scipy.stats import mode
from scipy.interpolate import interp1d



def Io():
    ans = raw_input("Already have coordinate files(Y)? Enter Y or N :")
    cols = ['objects','x1','y1','x2','y2']
    again = 1
    while again:
        if ans == 'Y' or ans == 'y' or ans =='':
            print("Choose coordinate file....")
            files = askopenfilename(initialdir = '/home/andyc/traffic_analysis/py/coordiante/')
            again = 0
        else:
            ans = raw_input("Manual key-in the coordinate(1)? Select them on the img(2)?  Enter 1 or 2 :")
            if ans == '1' or ans =='':            
                print("Enter the light 1 coordinate ...")
                L11 = []
                L12 = []
                L11.append(raw_input("X coordinate, upper left of traffic light 1:\n"))
                L11.append(raw_input("Y coordinate, upper left of traffic light 1:\n"))
                L12.append(raw_input("X coordinate, lower right of traffic light 1:\n"))
                L12.append(raw_input("Y coordinate, lower right of traffic light 1:\n"))

                L1 = np.array([L11,L12]).astype(int)
                print("Enter the light 2 coordinate ...")
                L21 = []
                L22 = []
                L21.append(raw_input("X coordinate, upper left of traffic light 2:\n"))
                L21.append(raw_input("Y coordinate, upper left of traffic light 2:\n"))
                L22.append(raw_input("X coordinate, lower right of traffic light 2:\n"))
                L22.append(raw_input("Y coordinate, lower right of traffic light 2:\n"))
                L2 = np.array([L21,L22]).astype(int)
                print("Enter the car coordinate ...")
                car1 = []
                car2 = []
                car1.append(raw_input("X coordinate, upper left of car:\n"))
                car1.append(raw_input("Y coordinate, upper left of car:\n"))
                car2.append(raw_input("X coordinate, lower right of car:\n"))
                car2.append(raw_input("Y coordinate, lower right of car:\n"))
                car = np.array([car1,car2]).astype(int)

                print("L1 is {0}\n".format(L1))
                print("L2 is {0}\n".format(L2))
                print("car is {0}\n".format(car))
                
                chk = raw_input("Is that right?(Y/n)\n")
                if chk == 'Y' or chk == 'y' or chk =='':
                    again = 0
                    print("Save the data ....\n")
                    filename = raw_input("input the file name (without extension):\n")+'.pkl'
                    print("where do you want to save?(choose folder)\n")
                    savepath = askdirectory(initialdir = '/home/andyc/traffic_analysis/py/coordiante/')
                    if savepath == '':
                        savepath = '/home/andyc/traffic_analysis/py/coordinate/'
                    else:
                        savepath = savepath+'/'

                    files = savepath+filename
                    cor ={}
                    cor['L1']=L1
                    cor['L2']=L2
                    cor['car']=car
                    pickle.dump(cor,open(files,"wb"),True)
                else:
                    print("Re-input the cooridnates....\n")
            else:  #select fome image
                opt = 0
                while (opt!='1'and opt!='2'):
                    opt = raw_input("Already has frames(1)?need to read from video(2)?  Enter 1 or 2 :")
                img = Read_frame(opt)
                tmpimg = np.zeros(img.shape) 
                tmpimg[:,:,:] = img
                figure(1),imshow(img)
              
                cor_result ='N'
                while (cor_result =='N' or cor_result =='n') :
                    print("Select first traffic light region.....")
                    L1 = np.round(ginput(2))
                    tmpimg[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],0] = 255  
                    tmpimg[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1] = 0
                    tmpimg[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],1] = 0
                    clf()
                    imshow(uint8(tmpimg))
                    cor_result = raw_input("This high light region(Red) is right?  Enter Y or N :")
                    if (cor_result =='N' or cor_result =='n') :
                        tmpimg[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],:] = img[L1[0][1]:L1[1][1],L1[0][0]:L1[1][0],:]
                    clf()
                    imshow(img)
                cor_result ='N'
                while (cor_result =='N' or cor_result =='n') :
                    print("Select Second traffic light region.....")
                    L2 = np.round(ginput(2))
                    tmpimg[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],0] = 255
                    tmpimg[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1] = 102
                    tmpimg[L2[0][1]:L2[1][1],L2[0][0]:L2[1][0],1] = 0
                    clf()
                    imshow(uint8(tmpimg))
                    cor_result = raw_input("This high light region(orange) is right?  Enter Y or N :")
                    clf()
                    imshow(img)
                cor_result ='N'
                while (cor_result =='N' or cor_result =='n') :
                    print("Select Car region.....")
                    car = np.round(ginput(2))
                    tmpimg[car[0][1]:car[1][1],car[0][0]:car[1][0],0] = 102
                    tmpimg[car[0][1]:car[1][1],car[0][0]:car[1][0],1] = 0
                    tmpimg[car[0][1]:car[1][1],car[0][0]:car[1][0],1] = 153
                    clf()
                    imshow(uint8(tmpimg))
                    cor_result = raw_input("This high light region(purple) is right?  Enter Y or N :")
                    clf()
                    imshow(img)
                again = 0    
                plt.close(figure(1))
                print("Save the data ....\n")
                filename = raw_input("input the file name (without extension):\n")+'.pkl'
                print("where do you want to save?(choose folder)\n")
                savepath = askdirectory(initialdir = '/home/andyc/traffic_analysis/py/coordinate/')
                if savepath == '':
                    savepath = '/home/andyc/traffic_analysis/py/coordinate/'
                else:
                    savepath = savepath+'/'

                files = savepath+filename
                cor ={}
                cor['L1']=L1
                cor['L2']=L2
                cor['car']=car
                pickle.dump(cor,open(files,"wb"),True)

    return files

def Read_frame(opt):
    if opt == '1': #Already has frames
        print("Choose frame file....")
        files = askopenfilename(initialdir = '/home/andyc/image/')
        frame = nd.imread(files)
    else: #read from video
        rval = False
        while not rval:
            print("Choose video file....")
            fname = askopenfilename(initialdir = '/home/andyc/Videos/')
            video = cv2.VideoCapture(fname)
            if video.isOpened():
                rval,frame = video.read()
            else:
                rval = False
        video.release()
    return frame

def Extract_frame():
    print("Choose video file....")
    fname = askopenfilename(initialdir = '/home/andyc/Videos/')
    print("where do you want to save?(choose folder)\n")
    savepath = askdirectory(initialdir = '/home/andyc/image/')
    while savepath == '':
        print("wrong folder!!\n")
        print("where do you want to save?(choose folder)\n")
        savepath = askdirectory(initialdir = '/home/andyc/image/')
    savepath = savepath+'/'
 
    video = cv2.VideoCapture(fname)
    if video.isOpened():
        rval,frame = video.read()
    else:
         rval = False
    idx = 0
    while rval:
          print("Extracting {0} frame".format(idx)) 
          rval,frame = video.read()
          name = savepath+str(idx).zfill(6)+'.jpg'
          if rval:
             cv2.imwrite(name,frame)
          idx = idx+1
    video.release()
    return savepath

def Tra_Ana(multi,path,pts_cor):

    imlist = sorted(glob.glob( os.path.join(path, '*.jpg')))
    nrow,ncol,nband = nd.imread(imlist[0]).shape

    dic = pickle.load(open(pts_cor,"rb"))

    L1 = dic['L1']
    L2 = dic['L2']
    car = dic['car']

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
            #print("file name 1: {0}".format(sub_imlist2[i]))
            #print("file name 2: {0}".format(sub_imlist[i]))

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

def Rm(idx,var,fps):
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
            if (idx[i]-idx[i-1])<4*fps:
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

def Trans_Idx(mean_mtx,var,fps): # mtx are both a N*3 arrays                                                                          

    RG = (mean_mtx[::,1]*mean_mtx[::,0])
    GR = (mean_mtx[::,1]*mean_mtx[::,0])
    v_RG = np.r_[mean_mtx[::,1]>0] & np.r_[mean_mtx[::,0]<0] \
                                   & np.r_[RG<-2]
    v_GR = np.r_[mean_mtx[::,1]<0] & np.r_[mean_mtx[::,0]>0] \
                                   & np.r_[GR<-2]
    RG_idx = np.array([i for i in range(len(mean_mtx)) if v_RG[i]==True])
    GR_idx = np.array([i for i in range(len(mean_mtx)) if v_GR[i]==True])

    return Rm(RG_idx,var[::,0],fps),Rm(GR_idx,var[::,0],fps)

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
def React_Time(mv_idx,RG_idx,mv_var,fps,partial=0.2): # mv_var is N*1 vector                                                          
    ck_th = fps*13
    mv_idx = np.asarray(mv_idx)
    mv_var = np.asarray(mv_var)
    RG_idx = np.asarray(RG_idx)
    Nsf = np.zeros(len(mv_idx))
    period = np.zeros(len(mv_idx))
    for ii in range(len(mv_idx)):
        print ii
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
                RB = [table[i] for i in range(len(tmp)) if tmp[i]==False][0]
                #RB = LB+1                                                                                                            
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
                    #RB = LB+1                                                                                                        
                    RB = [table[i] for i in range(len(tmp)) if tmp[i]==False][0]
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
    ptspath = Io()
  
    ans = raw_input("Doing data analysis ?  Enter Y or N :")    
    if ans == 'Y' or ans == 'y' or ans =='':

        ans = raw_input("Need extract frames from video?  Enter Y or N :")
        if ans == 'Y' or ans == 'y' or ans =='':
            impath = Extract_frame() 
        else:
            print("where are the frames?(choose folder)\n")
            impath = askdirectory(initialdir = '/home/andyc/image/')
            while impath == '':
                print("wrong folder!!\n")
                print("where are the frames?(choose folder)\n")   
                impath = askdirectory(initialdir = '/home/andyc/image/')
        

        L1_var,L2_var,car_var,env_var,L1_avg,L2_avg = Tra_Ana(16,impath,ptspath)
     
        print("Save the analysis data .... (choose folder)\n")
        savepath = askdirectory(initialdir = '/home/andyc/traffic_analysis/py/')
        while savepath == '':
            print("wrong folder!!\n")
            print("Save the analysis data .... (choose folder)\n")
            savepath = askdirectory()
        savepath = savepath +'/'
        name = savepath+'L1_var.pkl'
        pickle.dump(L1_var,open(name,"wb"),True)
        name = savepath+'L2_var.pkl'
        pickle.dump(L2_var,open(name,"wb"),True)
        name = savepath+'car_var.pkl'
        pickle.dump(car_var,open(name,"wb"),True)
        name = savepath+'env_var.pkl'
        pickle.dump(env_var,open(name,"wb"),True)
        name = savepath+'L1_avg.pkl'
        pickle.dump(L1_avg,open(name,"wb"),True)
        name = savepath+'L2_avg.pkl'
        pickle.dump(L2_avg,open(name,"wb"),True)
 
    else:      
        print("Where is analysis data? (choose folder)\n")
        datapath = askdirectory(initialdir = '/home/andyc/traffic_analysis/py/')
        while datapath == '':
            print("wrong folder!!\n")
            print("Where is analysis data?(choose folder)\n")
            datapath = askdirectory(initialdir = '/home/andyc/traffic_analysis/py/')
            datapath = datapath +'/'
        name = datapath+'L1_var.pkl'
        L1_var = pickle.load(open(name,"rb"))
        name = datapath+'L2_var.pkl'
        L2_var = pickle.load(open(name,"rb"))
        name = datapath+'car_var.pkl'
        car_var = pickle.load(open(name,"rb"))
        name = datapath+'env_var.pkl'
        env_var = pickle.load(open(name,"rb"))
        name = datapath+'L1_avg.pkl'
        L1_avg = pickle.load(open(name,"rb"))
        name = datapath+'L2_avg.pkl'
        L2_avg = pickle.load(open(name,"rb"))
    
    fps = int(raw_input("What is the fps of the video? (default is 30 fps)"))
    
    L1_RG_idx,L1_GR_idx = Trans_Idx(np.asarray(L1_avg.values()),np.asarray(L1_var.values()),fps)
    L2_RG_idx,L2_GR_idx = Trans_Idx(np.asarray(L2_avg.values()),np.asarray(L2_var.values()),fps)

    L1_VAR_RG = Trans_Var(np.asarray(L1_var.values()),L1_RG_idx)
    L1_VAR_GR = Trans_Var(np.asarray(L1_var.values()),L1_GR_idx)
    L2_VAR_RG = Trans_Var(np.asarray(L2_var.values()),L2_RG_idx)
    L2_VAR_GR = Trans_Var(np.asarray(L2_var.values()),L2_GR_idx)
    
    env_VAR = nd.gaussian_filter(np.asarray(env_var.values()),3) 

    car_VAR = nd.gaussian_filter(np.asarray(car_var.values()),3)
    lcmax_idx_R = Local_Max(car_VAR[::,0])

    react_T,Nsf = React_Time(Move_Idx(L1_RG_idx,lcmax_idx_R),L1_RG_idx,car_VAR[::,0],fps)

    env_VM= Bg_Ana(env_VAR[::,0],L1_RG_idx,np.round(Nsf))

 
    #plot figure

    figure(1,figsize=[7.5,7.5]),

    plot(range(len(L1_var)),L1_VAR_RG[::,0],color = '#990000',lw=2)
    plot(range(len(L1_var)),L2_VAR_RG[::,0], color = '#006600',lw=2)

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

    figure(5,figsize=[7.5,7.5]),
    plot(log(env_VM),react_T/fps,'ro')

    plt.grid(b=1,lw =2)
    plt.xlabel('Variance of background [arb units]')
    plt.ylabel('Time spend [s]')
    title('Relation between Driver Reaction Time and background variance')
#    title('Relation between Driver Reaction Time and ped variance')                                                                  
    for i in range(len(env_VM)):
        plt.text(log(env_VM[i]),react_T[i]/fps,i)

Main()
