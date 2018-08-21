# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:13:41 2018

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

def createtrainandtest(data,volume,start,number,pricewindow,volumewindow):
    trainX=[]
    trainY=[]
    testX=[]
    
   #testY=[]
    for i in range(number):
        temptr=data[start+i:start+i+pricewindow]
        tempvo=volume[start+i+pricewindow-volumewindow:start+i+pricewindow]
        temp1=np.hstack((np.array(temptr),np.array(tempvo)))
        trainX.append(temp1)
        trainY.append(data[start+i+pricewindow])
    test1=data[len(data)-pricewindow:len(data)]
    test2=volume[len(data)-volumewindow:len(data)]
    temp2=np.hstack((np.array(test1),np.array(test2)))
    testX.append(temp2)
   #testY.append(data[start+number+pricewindow])
    
    return trainX,trainY,testX
#data:close price vector
#volume:volume vector
#the size of training sample
#window1,2 how many price/volume data used in 1 sample
#bar1,2 base-----theta    
def cubicregression(data,volume,trainlength=240,window1=18,window2=3,bar1=0.05,bar2=0.03,base=0.0007):
    testX=[]
    trainX=[]
    trainY=[]
    
    start=len(data)-window1-trainlength
    trainX,trainY,testX=createtrainandtest(data,volume,start,trainlength,window1,window2)
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
   #testY=np.array(testY)
        #if trainY[-1]==testX[0,17] and testX[0,0:17].all()==trainX[-1,1:18].all():
            #print(1)
    theta=findtheta(testX[0][0:window1],bar1,bar2,base)
        
    tr3=trainX**3  
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
    testPredict = np.dot(Xt,w)
    prediction=testPredict[0]
    lastone=trainY[-1]
    if ((prediction-lastone)/lastone)>=theta:
        return(1)
    else:
        return(0)
        
        
#read data part
   
def readdata(csv_file):
    volume=[]
    close=[]
    for stu in csv_file:
        if 'Date' not in stu:
            volume.append(float(stu[6]))
            close.append(float(stu[5]))
    return volume,close    
pathbcc='D:/newdata/BCCUSDT_1H.csv'
patheth='D:/newdata/ETHUSDT_1H.csv'
pathltc='D:/newdata/LTCUSDT_1H.csv'
pathbtc='D:/newdata/BTCUSDT_1H.csv'

csv_bcc=csv.reader(open(pathbcc,'r'))
csv_eth=csv.reader(open(patheth,'r'))
csv_ltc=csv.reader(open(pathltc,'r'))
csv_btc=csv.reader(open(pathbtc,'r'))
bccv,bccc=readdata(csv_bcc)        
ethv,ethc=readdata(csv_eth)   
ltcv,ltcc=readdata(csv_ltc)   
btcv,btcc=readdata(csv_btc)   
bccv=bccv[1943:-1]
bccc=bccc[1943:-1]
ethv=ethv[3998:len(ethv)-33]
ethc=ethc[3998:len(ethc)-33]
ltcv=ltcv[1175:-1]
ltcc=ltcc[1175:-1]
btcv=btcv[4000:-1]
btcc=btcc[4000:-1]
print(cubicregression(ethc,ethv))