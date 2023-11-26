import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as data
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense , Dropout , LSTM
from tensorflow.keras.models import Sequential

start = dt.datetime(2010,1,1)
end = dt.datetime(2019,12,31)

df = yf.download("MSFT", start , end)
# print(df.head())
# print(df.tail())

# df = data.get_data_yahoo("AAPL" , start , end)
# df.head()

ma100 = df.Close.rolling(100).mean()

# plt.figure(figsize=(12,6))
# plt.plot(df.Close)
# plt.plot(ma100 , "r")
# plt.show()

ma200 = df.Close.rolling(200).mean()

plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100 , "r")
plt.plot(ma200 , "g")
plt.show()

# Splitting the data into training and testing

df_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
df_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

scaler = MinMaxScaler(feature_range = (0,1))
df_train_scale = scaler.fit_transform(df_train)


x_train = []
y_train = []

for i in  range(100 , df_train_scale.shape[0]):
    x_train.append(df_train_scale[i-100:i])
    y_train.append(df_train_scale[i,0])

x_train , y_train = np.array(x_train) , np.array(y_train)


# ML Model

model = Sequential()

model.add(LSTM(units = 50 , activation = 'relu' , return_sequences = True , input_shape = (x_train.shape[1] , 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 60 , activation = 'relu' , return_sequences = True))
model.add(Dropout(0.3))

model.add(LSTM(units = 50 , activation = 'relu' , return_sequences = True))
model.add(Dropout(0.4))

model.add(LSTM(units = 50 , activation = 'relu'))
model.add(Dropout(0.4))

model.add(Dense(units = 1))

# model.summry() Tells about the summry

#print(model.summary())

model.compile(optimizer = "adam" , loss = "mean_squared_error")
model.fit(x_train , y_train , epochs = 50)

model.save("Keras_model.h5")

df_test.head()

past_100_days = df_train.tail(100)

final_df = past_100_days.append(df_test , ignore_index = True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)

y_pred = model.predict(x_test)

scaler.scale_

scaler_factor = 1/0.02099517
y_pred = y_pred * scaler_factor
y_test = y_test * scaler_factor

plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label = "Orignal Price")
plt.plot(y_pred , 'r' , label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()