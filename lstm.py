# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:24:35 2018

@author: pxr13258
"""

import numpy as np
import pandas as pd
import csv
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
from keras.layers import Dropout

from keras.models import load_model



def create_dataset(dataset, look_back=1): 
    dataX, dataY = [], [] 
    for i in range(len(dataset)-look_back-1): 
        a = dataset[i:(i+look_back),0] 
        dataX.append(a) 
        dataY.append(dataset[i + look_back,0]) 
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
    #temp3.index = pd.to_datetime(temp3.index)19369
    #test3.append(temp3)
data = pd.read_csv("D:/dataset/LTCBTC_15T.csv",index_col= 'DetailTime')
train=[]
test=[]
test2=[]
test3=[]
for i in range(500):
    #emp=data[8111+i:8111+i+1000]
    temp=data[26900+i:26900+i+500]
    temp.index = pd.to_datetime(temp.index)
    train.append(temp)
   #temp1=data[8111+i+1000:8111+i+1001]
    temp1=data[26900+i+500:26900+i+501]
    temp1.index = pd.to_datetime(temp1.index)
    test.append(temp1)
   #temp2=data[8111+i+1000-20:8111+i+1000+50]
    temp2=data[26900+i+500:26900+i+500+80]
    temp2.index = pd.to_datetime(temp2.index)
    test2.append(temp2)
real80=[]
test80=[]
real1=[]
pre1=[]
pre11=[]
reallast=[]

for i in range(400,450):
    print(i)
    source1=train[i]
    target=test2[i]
    ts = source1['Close']
    ts1=target['Close']
    only1=test[i]['Close'][0]
    dataset=[]
    testset=[]
    for i in range(len(ts)):
        dataset.append(ts[i])
    for i in range(len(ts1)):
        testset.append(ts1[i])
    dataset=np.array(dataset)
    testset=np.array(testset)

#btc_trace = go.Scatter(x=data.index, y=data['Close'], name= 'Price')
    #plt.plot(testset[20:-1])
    main=testset[20:-1]
    real80.append(main)

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset=dataset.reshape(-1,1)
    testset=testset.reshape(-1,1)
#print(dataset)
    dataset = scaler.fit_transform(dataset)
    testset=scaler.fit_transform(testset)


    look_back = 20
    trainX, trainY = create_dataset(dataset, look_back)
    testX,testY= create_dataset(testset, look_back)
#print(trainX.shape[1])
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential() 
    #model.add(LSTM(128, input_shape=(1, look_back),return_sequences=True)) 
   # model.add(Dropout(0.2))
    model.add(LSTM(40,input_shape=(1, look_back)))
    #model.add(Dropout(0.2))
    model.add(Dense(1))
    #model.add(LSTM(4, input_shape=(1, look_back))) 
    #model.add(Dense(1)) 
    
    model.compile(loss='mean_squared_error', optimizer='adam') 
    model.fit(trainX, trainY, epochs=300, batch_size=40, verbose=0) 
    print('done')
#model.save('lookeback20_15minute.h5')
#model = load_model('lookeback20_15minute.h5')

    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
#print(trainPredict)
    trainPredict = scaler.inverse_transform(trainPredict) 
    trainY = scaler.inverse_transform([trainY]) 
    testPredict = scaler.inverse_transform(testPredict) 
    testY = scaler.inverse_transform([testY])
# shift train predictions for plotting 
    trainPredictPlot = np.empty_like(dataset) 
    trainPredictPlot[:, :] = np.nan 
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
#plt.plot(trainPredictPlot) 
    test80.append(testPredict[:,0])
    #plt.plot(testPredict[:,0])
    #plt.show()
    
    
    #ttest=test3[i]
    #ts3=ttest['Close']
    #ts3=np.array(ts3)
    #ts3=ts3.reshape(-1,1)
    #ts3=scaler.fit_transform(np.array(ts3))
    #ts3=ts3.reshape(1,1,20)
    #testone=model.predict(ts3)
    #testone=scaler.inverse_transform(testone) 
   # testone1=scaler.inverse_transform(model.predict(trainX[-1,0].reshape(1,1,20)))
    print(testPredict[0,0],only1,ts[-1])
    pre1.append(testPredict[0,0])
    reallast.append(ts[-1])
    real1.append(only1)
    
csvFile2 = open('lthbtc-450.csv','w', newline='') # 设置newline，否则两行之间会空一行

writer = csv.writer(csvFile2)

m = len(pre1)
writer.writerow(['pre1','real1','reallast'])
for i in range(m):

    writer.writerow([pre1[i],real1[i],reallast[i]])

csvFile2.close()