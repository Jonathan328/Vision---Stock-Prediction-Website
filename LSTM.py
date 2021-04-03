import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


df = web.DataReader('AAPL', data_source='yahoo',start='2012-01-01',end='2020-1-31') #Obtain Stock data from Yahoo Finance
df


#visualize the closing price history
plt.figure(figsize=(16,8))        
plt.title('Closing Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close Price USD ($)',fontsize = 18)
plt.show()

data = df.filter(['Close'])                  
dataset = data.values
training_data_len = math.ceil(len(dataset) * .8)
training_data_len
dataset

scaler = MinMaxScaler(feature_range= (0,1))       # Rescale the data so it could fit the input requirement of the LSTM model
scaled_data = scaler.fit_transform(dataset)
scaled_data


train_data = scaled_data[0:training_data_len,:]  
train_data
x_train = []                                      
y_train = []

print(train_data)

# Create training data set
for i in range(60,len(train_data)): 
    x_train.append(train_data[i-60:i])       # last 60 days data (input) 
    y_train.append(train_data[i,0])          # The predicted price of the next day (output) 
     
    
x_train,y_train = np.array(x_train),np.array(y_train)

x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1], 1)) # Reshape the data so it fits the LSTM model


model = Sequential()
model.add(LSTM(50,return_sequences = True, input_shape =(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs = 1)

test_data = scaled_data[training_data_len - 60: , :]
test_data
x_test = []
y_test = dataset[training_data_len: , :]
for i in range (60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
   

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
rmse = np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
valid['Predictions']



plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid['Predictions'])
plt.plot(valid['Close'])

plt.legend(['Train','Val','Predictions'],loc = 'lower right')
plt.show() 


# Predict stock prices at 1-2-2021
apple_quote = web.DataReader('AAPL',data_source = 'yahoo', start = '2012-01-01',end = '2020-1-31')
new_df = apple_quote.filter(['Close'])
last_60_days = new_df[-60:].values  
last_60_days_scaled = scaler.transform(last_60_days)
x_test = []
x_test.append(last_60_days_scaled)
x_test = np.array(x_test)
x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
pred_price = model.predict(x_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price 


