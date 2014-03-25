"""
install pandas in ubuntu:
 sudo apt-get install python-setuptools
 sudo easy_install pandas
"""

import csv,pickle
from tkFileDialog import askopenfilename
from tkFileDialog import askdirectory


def Io():
    ans = raw_input("Already have coordinate files(Y)? Enter Y or N :")
    cols = ['objects','x1','y1','x2','y2']
    again = 1
    while again:
        if ans == 'Y' or ans == 'y' or ans =='':  
            files = askopenfilename()
#            data = numpy.loadtxt(files, dtype=str, delimiter=';')
#            L1  = data[0][1::].astype(int)
#            L2  = data[1][1::].astype(int)
#            car = data[2][1::].astype(int)
            again = 0
        else: 
            print("Enter the light 1 coordinate ...")
            L1 = []
            L1.append(raw_input("X coordinate, upper left of traffic light 1:\n"))
            L1.append(raw_input("Y coordinate, upper left of traffic light 1:\n")) 
            L1.append(raw_input("X coordinate, lower right of traffic light 1:\n"))
            L1.append(raw_input("Y coordinate, lower right of traffic light 1:\n"))
            L1 = np.array(L1).astype(int)
            print("Enter the light 2 coordinate ...")
            L2 = []
            L2.append(raw_input("X coordinate, upper left of traffic light 2:\n"))
            L2.append(raw_input("Y coordinate, upper left of traffic light 2:\n"))
            L2.append(raw_input("X coordinate, lower right of traffic light 2:\n"))
            L2.append(raw_input("Y coordinate, lower right of traffic light 2:\n"))
            L2 = np.array(L2).astype(int)
            print("Enter the car coordinate ...")
            car = []
            car.append(raw_input("X coordinate, upper left of car:\n"))
            car.append(raw_input("Y coordinate, upper left of car:\n"))
            car.append(raw_input("X coordinate, lower right of car:\n"))
            car.append(raw_input("Y coordinate, lower right of car:\n"))
            car = np.array(car).astype(int)

            print("L1 is {0}\n".format(L1)) 
            print("L2 is {0}\n".format(L2))
            print("car is {0}\n".format(car))

            chk = raw_input("Is that right?(Y/n)\n")             
            if chk == 'Y' or chk == 'y' or chk =='':
                again = 0               
                print("Save the data ....\n")
                filename = raw_input("input the file name (without extension):\n")+'.pkl'
                print("where do you want to save?(choose folder)\n")
                savepath = askdirectory()
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

    return files   

Io()
