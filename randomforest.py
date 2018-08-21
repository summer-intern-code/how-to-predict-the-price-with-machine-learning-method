# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:06:15 2018

@author: pxr13258
"""

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor  
from sklearn.ensemble import RandomForestRegressor 
from sklearn.ensemble import RandomForestClassifier
# Load dataset

import csv


data = pd.read_csv("D:/dataset/ETHBTC_15T.csv",index_col= 'DetailTime')
#data.head(5)
#print(ts)
train=[]
test=[]
test2=[]
test3=[]

def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back),0] 
        dataX.append(a) 
        dataY.append((dataset[i + look_back,0]-dataset[i + look_back-1,0])/dataset[i + look_back-1,0]) 
    return np.array(dataX), np.array(dataY)
#for i in range(1000):
    #temp=data[len(data)-288*2-i-1:len(data)-288-i]
    #temp.index = pd.to_datetime(temp.index)
    #train.append(temp)
    #temp1=data[len(data)-288-i:len(data)-288-i+1]
    #temp1.index = pd.to_datetime(temp1.index)
    #test.append(temp1)
    #temp2=data[len(data)-288-i-20:len(data)-288-i+80]
    #temp2.index = pd.to_datetime(temp2.index)
    #test2.append(temp2)
    #temp3=data[len(data)-288-i-20:len(data)-288-i]
    #temp3.index = pd.to_datetime(temp3.index)
    #test3.append(temp3)25627
for i in range(10000):
    #emp=data[8111+i:8111+i+1000]
    temp=data[18900+i:18900+i+1000]
    temp.index = pd.to_datetime(temp.index)
    train.append(temp)
   #temp1=data[8111+i+1000:8111+i+1001]
    temp1=data[18900+i+1000:18900+i+1001]
    temp1.index = pd.to_datetime(temp1.index)
    test.append(temp1)
   #temp2=data[8111+i+1000-20:8111+i+1000+50]
    temp2=data[18900+i+1000-20:18900+i+1000+50]
    temp2.index = pd.to_datetime(temp2.index)
    test2.append(temp2)
real80=[]
test80=[]
real1=[]
pre1=[]
pre11=[]
reallast=[]

from scipy import optimize 
import math
def f_1(x, A, B):  
    return A*x + B


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
    if target<=0.00082:
        target=0.0001
    
    if target>0.00082:
        target=0.0015
    
        
    
    return target
#thetaset=[]
#for i in range(len(test2)):
    #thetas=findtheta(train[i][900:1000]['Close'])
    #thetaset.append(thetas)
    
    


#writer.writerow(['pre1','real1','reallast'])
#for i in range(m):

    #writer.writerow([pre1[i],real1[i],reallast[i]])

#csvFile2.close()

for i in range(100):
    print(i)
    source1=train[i]
    target=test2[i]
    ts = source1['Close']
    ts1=target['Close']
    only1=test[i]['Close'][0]
    #real1.append(only1)
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
    trainX, trainY = create_dataset(dataset, look_back)
    testX,testY= create_dataset(testset, look_back)
    trainX = scaler.fit_transform(trainX)
    testX=scaler.fit_transform(testX)


    look_back = 20
    rf=RandomForestClassifier()#这里使用了默认的参数设置  
    rf.fit(trainX,trainY)#
    prediction=rf.predict(testX)
    testPredict = scaler.inverse_transform([prediction])
    print(testPredict[0],testY[0])
    pre1.append(testPredict[0])
    test1.append(testY[0])

        