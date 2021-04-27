import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import statistics

import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error



new_df = pd.read_csv('xg_2017_01_final.csv')

X, y = new_df.iloc[:, :-1], new_df.iloc[:, -1]

X[:,0] = X[:,0] / 6

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)


'''
n_steps = X_train.shape[1]
n_features = 1


X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], n_features))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1], n_features))


model = Sequential()
model.add(LSTM(1), batch_input_shape(None, 6, 1), return_sequences=False)

model.compile(loss='mean_absolute_error', optimizer='adam',metrics=['accuracy'])

model.fit(X_train, y_train, epocs=100, validation_data=(X_test, y_test))


results = model.predict(X_test)

plt.scatter(range(20), results, c='r')
plt.scatter(range(20). y_test, c='g')
plt.show()


plt.plot(history.history['loss'])
plt.show()


print(X_test.shape)
print(X_train.shape)
print(y.shape)

'''