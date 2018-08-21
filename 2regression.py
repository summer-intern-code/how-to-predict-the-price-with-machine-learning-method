# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 13:26:06 2018

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
def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back),0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back,0]) 
    return np.array(dataX), np.array(dataY)
def findtheta(shuzu):
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
    if target<=0.0007:
        target=0.0001
    
    if target>0.0007:
        target=0.0015
    
        
    
    return target
def linearregression(datapath,start,length,trainlength,window):
    data = pd.read_csv(datapath,index_col= 'DetailTime')
    train=[]
    test=[]
    test2=[]
    test3=[]
    theta=[]
    for i in range(length):
    #emp=data[8111+i:8111+i+1000]
        temp=data[start+i:start+i+trainlength]
        temp.index = pd.to_datetime(temp.index)
        train.append(temp)
   #temp1=data[8111+i+1000:8111+i+1001]
        temp1=data[start+i+trainlength:start+i+trainlength+1]
        temp1.index = pd.to_datetime(temp1.index)
        test.append(temp1)
   #temp2=data[8111+i+1000-20:8111+i+1000+50]
        temp2=data[start+i+trainlength-window:start+i+trainlength+50]
        temp2.index = pd.to_datetime(temp2.index)
        test2.append(temp2)
    real80=[]
    test80=[]
    real1=[]
    pre1=[]
    pre11=[]
    reallast=[]
    
    for i in range(length):
        print(i)
        source1=train[i]
        target=test2[i]
        ts = source1['Close']
        ts1=target['Close']
        only1=test[i]['Close'][0]
        real1.append(only1)
        dataset=[]
        testset=[]
        for i in range(len(ts)):
            dataset.append(ts[i])
        for i in range(len(ts1)):
            testset.append(ts1[i])
        dataset=np.array(dataset)
        testset=np.array(testset)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset=dataset.reshape(-1,1)
        testset=testset.reshape(-1,1)
#print(dataset)
        dataset = scaler.fit_transform(dataset)
        testset=scaler.fit_transform(testset)


        look_back = window
        trainX, trainY = create_dataset(dataset, look_back)
        testX,testY= create_dataset(testset, look_back)
#print(trainX.shape[1])
   
        y_trainnorm=trainY
        y_testnorm=testY
        tr3=trainX**3
        tr4=trainX**4
        trainX=np.column_stack((trainX,trainX**2))
        trainX=np.column_stack((trainX,tr3))
        #trainX=np.column_stack((trainX,tr4))
        te3=testX**3
        te4=testX**4
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
        testPredict = scaler.inverse_transform([np.dot(Xt,w)]) 
        testY = scaler.inverse_transform([testY])
        pre1.append(testPredict[0,0])
        reallast.append(ts[-1])
        theta.append(findtheta(ts[0:20]))
    return pre1,real1,reallast,theta
pre,real,last,theta=linearregression('D:/dataset/LTCBTC_1H.csv',5155,1500,600,10)
plt.plot(real);plt.plot(pre)
plt.show()



            