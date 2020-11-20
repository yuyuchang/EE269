import numpy as np
import pandas as pd
from utils import *
from dataset import *
from evaluate import *
from constant import *
import argparse
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D, GRU, LeakyReLU, LSTM, TimeDistributed, dot, Reshape, multiply, Concatenate, RepeatVector, Flatten, BatchNormalization, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers

parser = argparse.ArgumentParser(description='split data to training, validation and testing set')
parser.add_argument('--data', type=str, default='../../data/kdd2018/london.csv')
parser.add_argument('--weather_data', type=str, default='../../data/kdd2018/ld_hourly_worldweather.csv')
parser.add_argument('--input_length', type=int, default=24)
args = parser.parse_args()

df_aq = pd.read_csv(args.data).drop_duplicates(subset = ['stationId', 'utc_time'], keep = 'last')
df_weather = pd.read_csv(args.weather_data)
df_weather = df_weather[['stationId', 'utc_time', 'tempC', 'windspeedKmph', 'precipMM', 'humidity', 'pressure', 'cloudcover', 'HeatIndexC', 'WindChillC', 'WindGustKmph', 'wind_cos', 'wind_sin', 'wind_flow_cos', 'wind_flow_sin']]
stationId = citymeta['london']['schema']['station']
area = citymeta['london']['schema']['area']
station2area = citymeta['london']['station2area']
drop = ['year', 'quarter', 'month', 'week', 'weekday', 'date', 'day', 'hour', 'Unnamed: 0']
df_aq = tranform_corordinate(df_aq).drop(drop, axis = 1)
df = pd.merge(df_aq, df_weather, on = ['stationId', 'utc_time']).drop_duplicates(['stationId', 'utc_time'], keep = 'last')

print(df.head())
df = df.groupby('area')

X_train = None
X_valid = None
X_test = None
y_train = None
y_valid = None
y_test = None

test_id = []

for a in area:
  print(a)
  df_tmp = df.get_group(a).groupby('stationId')
  for s in stationId:
    if station2area[s] == a:
      print(s)
      test_id.append(s)
      data = df_tmp.get_group(s).sort_values('utc_time')
      data = data.drop(['stationId', 'area', 'utc_time'], axis = 1)
      labels = data[['PM2.5', 'PM10']]
      data = data.values
      labels = labels.values
      onehot = np.zeros((data.shape[0], len(area)))
      onehot[:, area.index(a)] = 1
      data = np.concatenate((data, onehot), axis = 1)
      train_tmp = data[:10176]
      valid_tmp = data[10176 - args.input_length:11640]
      test_tmp = data[11640 - args.input_length:12384]
      l_train_tmp = labels[:10176]
      l_valid_tmp = labels[10176 - args.input_length:11640]
      l_test_tmp = labels[11640 - args.input_length:12384]
      print(train_tmp)
      print(valid_tmp)
      print(test_tmp)
      X_train_tmp = np.zeros((train_tmp.shape[0] - args.input_length - 48 + 1, args.input_length * train_tmp.shape[1]))
      X_valid_tmp = np.zeros((valid_tmp.shape[0] - args.input_length - 48 + 1, args.input_length * valid_tmp.shape[1]))
      X_test_tmp = np.zeros((test_tmp.shape[0] - args.input_length - 48 + 1, args.input_length * test_tmp.shape[1]))
      y_train_tmp = np.zeros((train_tmp.shape[0] - args.input_length - 48 + 1, 48 * 2))
      y_valid_tmp = np.zeros((valid_tmp.shape[0] - args.input_length - 48 + 1, 48 * 2))
      y_test_tmp = np.zeros((test_tmp.shape[0] - args.input_length - 48 + 1, 48 * 2))

      for i in range(X_train_tmp.shape[0]):
        X_train_tmp[i] = np.reshape(train_tmp[i:i + args.input_length, :],(1, -1))
        y_train_tmp[i] = np.reshape(l_train_tmp[i + args.input_length: i + args.input_length + 48, :], (1, -1))
      for i in range(X_valid_tmp.shape[0]):
        X_valid_tmp[i] = np.reshape(valid_tmp[i:i + args.input_length, :],(1, -1))
        y_valid_tmp[i] = np.reshape(l_valid_tmp[i + args.input_length: i + args.input_length + 48, :], (1, -1))
      for i in range(X_test_tmp.shape[0]):
        X_test_tmp[i] = np.reshape(test_tmp[i:i + args.input_length, :],(1, -1))
        y_test_tmp[i] = np.reshape(l_test_tmp[i + args.input_length: i + args.input_length + 48, :], (1, -1))

      print(X_train_tmp)
      print(X_valid_tmp)
      print(X_test_tmp)
      print(y_train_tmp)
      print(y_valid_tmp)
      print(y_test_tmp)

      if X_train is None:
        X_train = X_train_tmp
        X_valid = X_valid_tmp
        X_test = X_test_tmp
        y_train = y_train_tmp
        y_valid = y_valid_tmp
        y_test = y_test_tmp
      else:
        X_train = np.concatenate((X_train, X_train_tmp))
        X_valid = np.concatenate((X_valid, X_valid_tmp))
        X_test = np.concatenate((X_test, X_test_tmp))
        y_train = np.concatenate((y_train, y_train_tmp))
        y_valid = np.concatenate((y_valid, y_valid_tmp))
        y_test = np.concatenate((y_test, y_test_tmp))

print(X_train.shape)
print(y_train.shape)
print(test_id)

X_train = np.reshape(X_train, (X_train.shape[0] * args.input_length, -1))
X_valid = np.reshape(X_valid, (X_valid.shape[0] * args.input_length, -1))
X_test = np.reshape(X_test, (X_test.shape[0] * args.input_length, -1))
y_train = np.reshape(y_train, (-1 ,2))
y_valid = np.reshape(y_valid, (-1, 2))
y_test = np.reshape(y_test, (-1, 2))

MAX = np.reshape(np.amax(X_train, axis = 0), (1, -1))
Y_MAX = np.reshape(np.array([MAX[0, 0], MAX[0, 1]]), (1, -1))

X_train /= MAX
X_valid /= MAX
X_test /= MAX
y_train /= Y_MAX
y_valid /= Y_MAX
y_test /= Y_MAX


X_train = np.reshape(X_train, (-1, args.input_length * X_train.shape[1]))
X_valid = np.reshape(X_valid, (-1, args.input_length * X_valid.shape[1]))
X_test = np.reshape(X_test, (-1, args.input_length * X_test.shape[1]))
y_train = np.reshape(y_train, (-1, 48 * 2))
y_valid = np.reshape(y_valid, (-1, 48 * 2))
y_test = np.reshape(y_test, (-1, 48 * 2))

x = Input((X_train.shape[1],))
y = Dense(y_train.shape[1])(x)

opt = optimizers.RMSprop(lr = 0.0001)
model = Model(inputs = x, outputs = y)
model.compile(loss = 'mae', optimizer = opt, metrics = ['mse', 'mae'])
filepath = 'ld_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', save_weights_only = False, save_best_only = True)
earlystop = EarlyStopping(monitor = 'val_loss', patience = 5)
callbacks_list = [checkpoint, earlystop]
print(model.summary())
model.fit(X_train, y_train, epochs = 5, batch_size = 1024, validation_data     = [X_valid, y_valid], callbacks = callbacks_list)
model = load_model(filepath)
predictions = model.predict(X_test, batch_size = 1024)

print(predictions.shape)

predictions = np.reshape(predictions, (-1, 2))
predictions *= Y_MAX
predictions = np.reshape(predictions, (len(test_id), -1, 2 * 48))

filename = 'output.csv'

f = open(filename, "a+")

for ID in range(13):
  for date in range(1, 31, 1):
    for hour in range(48):
      if date < 10:
        f.write("2018-05-0" + str(date) + "," + test_id[ID].split('_')[0][:10] + "_aq#" + str(hour) + "," + str(predictions[ID, 24 * (date - 1), 2 * hour]) + "," + str(predictions[ID, 24 * (date - 1), 2 * hour + 1]) + ",0\n")
      else:
        f.write("2018-05-" + str(date) + "," + test_id[ID].split('_')[0][:10] + "_aq#" + str(hour) + "," + str(predictions[ID, 24 * (date - 1), 2 * hour]) + "," + str(predictions[ID, 24 * (date - 1), 2 * hour + 1]) + ",0\n")

