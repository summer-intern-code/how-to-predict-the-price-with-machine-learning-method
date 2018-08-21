# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 09:45:11 2018

@author: pxr13258
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
# Load dataset
import math
import csv
pathbcc='D:/newdata/BCCUSDT_1H.csv'
patheth='D:/newdata/ETHUSDT_1H.csv'
pathltc='D:/newdata/LTCUSDT_1H.csv'
pathbtc='D:/newdata/BTCUSDT_1H.csv'

csv_bcc=csv.reader(open(pathbcc,'r'))
csv_eth=csv.reader(open(patheth,'r'))
csv_ltc=csv.reader(open(pathltc,'r'))
csv_btc=csv.reader(open(pathbtc,'r'))

def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back),0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back,0]) 
    return np.array(dataX), np.array(dataY)

def readdata(csv_file):
    volume=[]
    close=[]
    for stu in csv_file:
        if 'Date' not in stu:
            volume.append(float(stu[6]))
            close.append(float(stu[5]))
    return volume,close

def createtrainandtest(data,volume,start,number,pricewindow,volumewindow):
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    for i in range(number):
        temptr=data[start+i:start+i+pricewindow]
        tempvo=volume[start+i+pricewindow-volumewindow:start+i+pricewindow]
        temp1=np.hstack((np.array(temptr),np.array(tempvo)))
        trainX.append(temp1)
        trainY.append(data[start+i+pricewindow])
    test1=data[start+number:start+number+pricewindow]
    test2=volume[start+number+pricewindow-volumewindow:start+number+pricewindow]
    temp2=np.hstack((np.array(test1),np.array(test2)))
    testX.append(temp2)
    testY.append(data[start+number+pricewindow])
    
    return trainX,trainY,testX,testY

        
def findtheta(shuzu,bar1,bar2,base):
   #print(shuzu.type)
    mi=float(min(shuzu))
    ma=float(max(shuzu))
    xrey=np.arange(mi,ma,float((ma-mi)/len(shuzu)))
    if len(xrey)!=len(shuzu):
         xrey=list(xrey)
         del xrey[-1]
         xrey=np.array(xrey)
 
    [A,B]= np.polyfit(xrey, shuzu, 1)
    
    ang=math.atan(A)
    target=(0.5-float(ang/1.6)*0.6)*0.001
    #print(target,0)
    if target<=base:
        target=bar1
        
    else:
        target=bar2
    return target


bccv,bccc=readdata(csv_bcc)        
ethv,ethc=readdata(csv_eth)   
ltcv,ltcc=readdata(csv_ltc)   
btcv,btcc=readdata(csv_btc)   
bccv=bccv[1943:-1]
bccc=bccc[1943:-1]
ethv=ethv[3998:-1]
ethc=ethc[3998:-1]
ltcv=ltcv[1175:-1]
ltcc=ltcc[1175:-1]
btcv=btcv[4000:-1]
btcc=btcc[4000:-1]

def linearregression(data,volume,start,length,trainlength,window1,window2,bar1,bar2,base):
    pre1=[]
    real1=[]
    theta=[]
    for j in range(length):
        trainX,trainY,testX,testY=createtrainandtest(data,volume,start+j,trainlength,window1,window2)
        trainX=np.array(trainX)
        trainY=np.array(trainY)
        testX=np.array(testX)
        testY=np.array(testY)
        #if trainY[-1]==testX[0,17] and testX[0,0:17].all()==trainX[-1,1:18].all():
            #print(1)
        theta.append(findtheta(testX[0][0:window1],bar1,bar2,base))
        
        tr3=trainX**3
        #tr4=trainX**4
        trainX=np.column_stack((trainX,trainX**2))
        trainX=np.column_stack((trainX,tr3))
        #trainX=np.column_stack((trainX,tr4))
        te3=testX**3
        #te4=testX**4
        testX=np.column_stack((testX,testX**2))
        testX=np.column_stack((testX,te3))
        #testX=np.column_stack((testX,te4))
        one=np.ones((len(trainX[:,1]),1))
        X=np.column_stack((one,trainX))
        xtran=np.transpose(X)
        xdot=np.dot(xtran,X)
        xinv=np.linalg.inv(xdot)
        xddot=np.dot(xinv,xtran)
        w=np.dot(xddot,trainY)

        one=np.ones((len(testX[:,1]),1))
        Xt=np.column_stack((one,testX))
        #delta=np.dot(Xt,w)-testY
        testPredict = np.dot(Xt,w)
        testY = testY
        pre1.append(testPredict[0])
        real1.append(testY[0])
        
    return pre1,real1,theta

        
pre,real,theta=linearregression(ethc,ethv,0,2150,240,18,3,0.05,0.03,0.0007)
#lt.plot(real);plt.plot(pre)



