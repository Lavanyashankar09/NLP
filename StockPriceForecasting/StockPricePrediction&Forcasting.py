### Data Collection with tiingo
import pandas_datareader as pdr
import numpy as np
key="591db9e02cc577b32bfc5780b8f099a968c65280"
df = pdr.get_data_tiingo('AAPL', api_key=key)
df.to_csv('AAPL.csv')

#got csv file
import pandas as pd
df=pd.read_csv('AAPL.csv')
df2=df.reset_index()['close']

import matplotlib.pyplot as plt
plt.plot(df2)

#scaling
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df2).reshape(-1,1))

import matplotlib.pyplot as plt
plt.plot(df1)

##splitting dataset into train and test split
training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data=df1[0:training_size,:]
test_data=df1[training_size:len(df1),:1]

import numpy
# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return numpy.array(dataX), numpy.array(dataY)

# reshape into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 100
X_train_ts, y_train_ts = create_dataset(train_data, time_step)
X_test_ts, y_test_ts = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train_ts.reshape(X_train_ts.shape[0],X_train_ts.shape[1] , 1)
X_test = X_test_ts.reshape(X_test_ts.shape[0],X_test_ts.shape[1] , 1)

### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')
model.summary()

model.fit(X_train,y_train_ts,validation_data=(X_test,y_test_ts),epochs=100,batch_size=64,verbose=1)

### Lets Do the prediction and check performance metrics
y_train_pred_before=model.predict(X_train)
y_test_pred_before=model.predict(X_test)

##Transformback to original form
y_train_pred=scaler.inverse_transform(y_train_pred_before)
y_test_pred=scaler.inverse_transform(y_test_pred_before)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
mse_train = math.sqrt(mean_squared_error(y_train_ts,y_train_pred_before))
mse_test = math.sqrt(mean_squared_error(y_test_ts,y_test_pred_before))


### Plotting 
# shift train predictions for plotting
look_back=100
trainPredictPlot = numpy.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(y_train_pred_before)+look_back, :] = y_train_pred_before
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(df1)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(y_train_pred_before)+(look_back*2)+1:len(df1)-1, :] = y_test_pred_before
# plot baseline and predictions
plt.plot(df1)
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# demonstrate prediction for next 30 days
x_input=test_data[340:].reshape(1,-1)
temp_input_before=list(x_input)
temp_input=temp_input_before[0].tolist()

lst_output=[]
n_steps=100
i=0
j=30
while(i<j):
    
    if(len(temp_input)>n_steps):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print("the y hat value is ",yhat[0])
        temp_input.extend(yhat[0].tolist())
        print("the temp_input lenght is ",len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
print("the output for 30 days is ",lst_output)

day_new=np.arange(1,101)
day_pred=np.arange(101,131)
plt.plot(day_new,scaler.inverse_transform(df1[1157:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
df3=df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
df3=scaler.inverse_transform(df3).tolist()
plt.plot(df3)
