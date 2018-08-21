# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 13:37:14 2018

@author: pxr13258
"""

from hmmlearn.hmm import GaussianHMM 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from matplotlib import cm 
import matplotlib.pyplot as plt
import pickle

long=10
modeltype=6
patheth='D:/newdata/ETHUSDT_1H.csv'
eth= pd.read_csv(patheth)
openn=eth['Open'].values#[500:1000]
close=eth['Close'].values#[500:1000]
high=eth['High'].values#[500:1000]
low=eth['Low'].values#[500:1000]
Vol=eth['TotalVolume'].values#[0:500]
logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[long-1:] 
logreturn12 = np.log(np.array(close[long:]))-np.log(np.array(close[:-long])) 
diffreturn = (np.log(np.array(high))-np.log(np.array(low)))[long:] 
volidx= Vol[long:]
closeidx = close[long:]
X = np.column_stack([logreturn,logreturn12,diffreturn])#,volidx,closeidx]) 
#hmm = GaussianHMM(n_components = modeltype, covariance_type='diag',n_iter = 9000).fit(X)
#output = open('D:/hmmmodel.pkl', 'wb')
#s = pickle.dump(hmm, output)
#output.close()
inputt = open('D:/hmmmodel.pkl', 'rb')
hmm = pickle.load(inputt)
inputt.close()
latent_states_sequence = hmm.predict(X)
def makeindex(series):
    number=0
    num=[]
    for i in range(len(series)):
        num.append(number)
        number=number+1
    return num
index=makeindex(close)
#for i in range(len(latent_states_sequence)):
#    if latent_states_sequence[i]==0:
#        plt.scatter(index[i+long],close[i+long],10,color='red')
#    if latent_states_sequence[i]==1:
#        plt.scatter(index[i+long],close[i+long],10,color='green')
#    if latent_states_sequence[i]==2:
#        plt.scatter(index[i+long],close[i+long],10,color='blue')
#    if latent_states_sequence[i]==4:
#        plt.scatter(index[i+12],close[i+12],10,color='yellow')
#    if latent_states_sequence[i]==2:
#        plt.scatter(index[i+12],close[i+12],10,color='orange')
#    if latent_states_sequence[i]==2:
#        plt.scatter(index[i+12],close[i+12],10,color='black')
#plt.show()


index=index[long:]
data=pd.DataFrame({'index':index,'logreturn':logreturn,'logreturn12':logreturn12,'diffreturn':diffreturn})#,#})
for i in range(modeltype):
    state = (latent_states_sequence == i)
    idx = np.append(0,state[:-1])
    data['state %d_return'%i] = data.logreturn.multiply(idx,axis = 0) 
    k=i
    if k==0:
        clr='blue'#sell#
    if k==1:
        clr='red'
    if k==2:
        clr='green'#buy)
    if k==3:
        clr='orange'#sell#
    if k==4:
        clr='black'#buy#
    if k==5:
        clr='navy'
            
    plt.plot(np.exp(data['state %d_return' %i].cumsum()),color=clr)
pos=0 
action=0
poslist=[]
actionlist=[]  
#trade strategy0
for i in range(len(close)):
    if i<long:
        action=0
        pos=0
    else:
        if latent_states_sequence[i-long]==4 or latent_states_sequence[i-long]==2:
            action=1
#        elif latent_states_sequence[i-long]==3 or latent_states_sequence[i-long]==0 :
#            action=-sum(actionlist)
        else:
            action=-sum(actionlist)
    
    actionlist.append(action)
    pos=sum(actionlist)
    poslist.append(pos)
## trade strategy2
##for i in range(len(close)):
##    if i<long:
##        action=0
##        pos=0
##    else:
##        if poslist[-1]==0:
##           if latent_states_sequence[i-long]==4 or latent_states_sequence[i-long]==2:
##              action=1
###        elif latent_states_sequence[i-long]==3 or latent_states_sequence[i-long]==0 :
###            action=-sum(actionlist)
##           else:
##               action=0
##        elif poslist[-1]==1:
##            if latent_states_sequence[i-long]!=4 and latent_states_sequence[i-long]!=2:
##               action=-1
##            else:
##                action=0
##        else:
##            action=0
##             
##    
##    actionlist.append(action)
##    pos=sum(actionlist)
##    poslist.append(pos)
#
def cashcal(close,xamount):
    cash=0
    tolist=[]
    cashlist=[]
    po=[]
    for i in range(len(close)):
          cashlist.append(cash)
          positionx=sum(xamount[0:i+1])
          po.append(positionx)
          if xamount[i]!=0:
             cash=cash-xamount[i]*close[i]#-0.0005(abs(xamount[i]*x[i])+abs(yamount[i]*y[i]))
          total=cash+positionx*close[i]
          tolist.append(total)
          
    return tolist,cashlist,po
total,cash,pose=cashcal(close,actionlist)