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
import streamlit as st
from keras.models import load_model

start = dt.datetime(2010,1,1)
end = dt.datetime(2019,12,31)

st.title("ASAP")

user_input = st.text_input("Enter the Stock name : ")

df = yf.download(user_input, start , end)

st.subheader(user_input , " Date")

st.write(df.describe())

st.subheader("Closing Price vs Time Chart with 100 MA(Moving Average)")
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100 , "r")
st.pyplot(fig)

st.subheader("Closing Price vs Time Chart with 100 & 200 MA(Moving Average)")
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100 , "r")
plt.plot(ma200 , "g")
st.pyplot(fig)

# Splitting the data into x_train , y_train
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

# Load the model

model = load_model("Keras_model.h5")

# testing the model

past_100_days = df_train.tail(100)

final_df = past_100_days._append(df_test , ignore_index = True)

input_data = scaler.fit_transform(final_df)
x_test = []
y_test = []

for i in range(100 , input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)
y_pred = model.predict(x_test)

sc = scaler.scale_

scaler_factor = 1/sc[0]
y_pred = y_pred * scaler_factor
y_test = y_test * scaler_factor

# Final Graph

st.subheader("Prediction vs Orignal")

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test , 'b' , label = "Orignal Price")
plt.plot(y_pred , 'r' , label = "Predicted Price")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)