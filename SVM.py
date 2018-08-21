# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:45:29 2018

@author: pxr13258
"""
from sklearn import cross_validation  
from sklearn import preprocessing 
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import math
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

from sklearn import svm 

def readdata(csv_file):
    volume=[]
    close=[]
    openn=[]
    high=[]
    low=[]
    fluc=[]
    rate=[]
    target=[]
    for stu in csv_file:
        if 'Date' not in stu:
            volume.append(float(stu[6]))
            close.append(float(stu[5]))
            openn.append(float(stu[2]))
            high.append(float(stu[3]))
            low.append(float(stu[4]))      
            fluc.append(float(stu[12]))
            rate.append(float(stu[13]))
            target.append(stu[14])
    return volume,close,openn,high,low,fluc,rate,target

patheth='D:/newdata/ETHUSDT_1H.csv'
csv_eth=csv.reader(open(patheth,'r'))
ethv,ethc,etho,ethh,ethl,ethf,ethr,etht=readdata(csv_eth)   
ethv=ethv[3998:-1]
ethc=ethc[3998:-1]
etho=etho[3998:-1]
ethh=ethh[3998:-1]
ethl=ethl[3998:-1]
ethf=ethf[3998:-1]
ethr=ethr[3998:-1]
etht=etht[3998:-1]



def dataforsvm(volume,close,fluc,rate,target,start,number,length):
    trainX=[]
    trainY=[]
    testX=[]
    testY=[]
    for i in range(number):
        vol=volume[start+i:start+i+length]
        clo=close[start+i+length-1]
        flu=fluc[start+i:start+i+length]
        rat=rate[start+i:start+i+length]
        temp1=np.hstack((np.array(vol),np.array(flu)))
        temp2= np.hstack((temp1,np.array(rat)))
        trainX.append(np.hstack((temp2,np.array(clo))))
        trainY.append(target[start+i+length])
    vol1=volume[start+number:start+number+length]
    clo1=close[start+number+length-1]
    flu1=fluc[start+number:start+number+length]
    rat1=rate[start+number:start+number+length]
    temp11=np.hstack((np.array(vol1),np.array(flu1)))
    temp21= np.hstack((temp11,np.array(rat1)))
    testX.append(np.hstack((temp21,np.array(clo1))))
    testY.append(target[start+number+length])   
    return trainX,trainY,testX,testY
#trX,trY,teX,teY=dataforsvm(ethv,ethc,ethf,ethr,etht,0,10,3)

def foresttest(volume,close,fluc,rate,target,number,trainlength,totallength,nn):
    start=0
    pre=[]
    real=[]
    buy=0
    win=0
    for i in range(totallength):
        #print(i)
        trX,trY,teX,teY=dataforsvm(volume,close,fluc,rate,target,start+i,number,trainlength)
        myforest =  RandomForestClassifier(random_state=90, n_estimators=nn,criterion='gini',min_samples_split=3)

        myforest.fit(trX,trY)
        #trpre=clf_svm.predict(trX)
        prediction=myforest.predict(teX)
        pre.append(prediction)
        real.append(teY[0])
        if prediction[0]=='TRUE':
            buy=buy+1
            if teY[0]=='TRUE':
                win=win+1
    return pre,real,buy,win
for state in [10,30,50,70,100,130,160,190]:
    pre1,real1,buy,win=foresttest(ethv,ethc,ethf,ethr,etht,120,3,2200,state)
    print(buy,win,state,float(win/buy))
#50
