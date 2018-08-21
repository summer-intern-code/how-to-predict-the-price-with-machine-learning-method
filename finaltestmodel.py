# -*- coding: utf-8 -*-
"""
Created on Wed May 23 18:18:36 2018

@author: pxr13258
"""
import numpy as np
import pandas as pd
import seaborn;seaborn.set()
from sklearn.metrics import mean_squared_error
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15,6
import math
import plotly
plotly.offline.init_notebook_mode()
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

from keras.models import load_model
data = pd.read_csv("D:/dataset/binance_btc_15T.csv",index_col= 'DetailTime')
train=[]
test=[]
test2=[]
test3=[]
for i in range(1000):
    temp=data[len(data)-2880*2-i-1:len(data)-2880-i]
    temp.index = pd.to_datetime(temp.index)
    train.append(temp)
    temp3=data[len(data)-2880-i-20:len(data)-2880-i]
    temp3.index = pd.to_datetime(temp3.index)
    test3.append(temp3)
    temp1=data[len(data)-2880-i:len(data)-2880-i+1]
    temp1.index = pd.to_datetime(temp1.index)
    test.append(temp1)
    temp2=data[len(data)-2880-i:len(data)-2880-i+100]
    temp2.index = pd.to_datetime(temp2.index)
    test2.append(temp2)
def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back),0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back,0]) 
    return np.array(dataX), np.array(dataY)  
realone=[]
realtest=[]
realpredict=[]
preone=[]
reallast=[]
for i in range(len(train)):
    print(i)
    source1=train[i]
    target=test2[i]
    ttest=test3[i]
    only1=test[i]['Close'][0]
    realone.append(only1)
    #print(only1,target[1])
    ts = source1['Close']
    ts1=target['Close']
    ts3=ttest['Close']
    dataset=[]
    testset=[]
    for j in range(len(ts)):
        dataset.append(ts[j])
    for k in range(len(ts1)):
        testset.append(ts1[k])
    dataset=np.array(dataset)
    testset=np.array(testset)
    ts3=np.array(ts3)
#btc_trace = go.Scatter(x=data.index, y=data['Close'], name= 'Price')
    plt.plot(testset[19:-2])
    main=testset[19:-2]
    realtest.append(main)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset=dataset.reshape(-1,1)
    testset=testset.reshape(-1,1)
    ts3=ts3.reshape(-1,1)
#print(dataset)
    dataset = scaler.fit_transform(dataset)
    testset=scaler.fit_transform(testset)
    ts3=scaler.fit_transform(np.array(ts3))


    look_back = 20
    trainX, trainY = create_dataset(dataset, look_back)
    
    testX,testY= create_dataset(testset, look_back)
#print(trainX.shape[1])
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential() 
    model.add(LSTM(4, input_shape=(1, look_back))) 
    model.add(Dense(1)) 
    model.compile(loss='mean_squared_error', optimizer='adam') 
    model.fit(trainX, trainY, epochs=100, batch_size=10, verbose=2) 
    #print(i)
    print('Finished')
   #model.save('lookeback20_15minute.h5')
   #model = load_model('lookeback20_15minute.h5')

    #trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
#print(trainPredict)
    ts3=ts3.reshape(1,1,20)
    testone=model.predict(ts3)
    testone=scaler.inverse_transform(testone) 
    #tx=scaler.inverse_transform(trainX[-1,0,-1])
    print(testone,only1,ts[-1])
    preone.append(testone)
    reallast.append(ts[-1])
    testPredict = scaler.inverse_transform(testPredict) 
    #testY = scaler.inverse_transform([testY])
# shift train predictions for plotting 
    #trainPredictPlot = np.empty_like(dataset) 
    #trainPredictPlot[:, :] = np.nan 
    #trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#plt.plot(trainPredictPlot) 

    plt.plot(testPredict[:,0])
    realpredict.append(testPredict[:,0])
    plt.show()
    #......pretest=model.predict(train[-1]) preset.append(pretest[......])....1point test
    #testpredict (print(rate1,2,3,...))
    #.......1000pointtest
    