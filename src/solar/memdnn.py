import argparse
import numpy as np
import pandas as pd
import pickle
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import keras.backend as K
import tensorflow as tf
from keras.models import Sequential, Model, load_model
from keras.layers.embeddings import Embedding
from keras.layers import Input, Activation, Dense, Permute, Dropout, add, dot, concatenate, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, AveragePooling1D, AveragePooling2D, GRU, LeakyReLU, LSTM, TimeDistributed, dot, Reshape, multiply, Concatenate, RepeatVector, Flatten, BatchNormalization, merge
from keras.callbacks import EarlyStopping, ModelCheckpoint
import sys
from keras import optimizers
import math
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from utils import *
from keras.layers.core import *
from attention_utils import get_activations, get_data_recurrent

SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs, timesteps):
  input_dim = int(inputs.shape[2])
  a = Permute((1, 3, 2))(inputs)
  a = Dense(timesteps, activation = 'softmax')(a)
  if SINGLE_ATTENTION_VECTOR:
    a = Lambda(lambda x: K.mean(x, axis = 1), name = 'dim_reduction')(a)
    a = RepeatVector(input_dim)(a)
  a_probs = Permute((1, 3, 2))(a)
  output_attention_mul = merge([inputs, a_probs], mode = 'mul')
  return output_attention_mul

gpu_options = tf.GPUOptions(allow_growth = True)
sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
tf.keras.backend.set_session(sess)

parser = argparse.ArgumentParser(description='MEMRNN Time series prediction')
parser.add_argument('--data', type=str, default='../../data/solar-energy/solar_AL.txt', help='location of the data file')
parser.add_argument('--CNN_unit', type=int, default=32, help='number of CNN hidden unit')
parser.add_argument('--GRU_unit', type=int, default=32, help='number of GRU hidden unit')
parser.add_argument('--CNN_kernel', type=int, default=2, help='kernel size of CNN')
parser.add_argument('--input_length', type=int, default=6, help='number of hours')
parser.add_argument('--horizon', type=int, default=3, help='horizon')
parser.add_argument('--AR', type=int, default=1, help='AR')
#parser.add_argument('--save', type=str, default='model/model.h5', help='pat    h to save model')
args = parser.parse_args()

df = pd.read_csv(args.data, header = None)
df = df.values
sz = df.shape[0]
feature_sz = df.shape[1]
df_train = df[:int(sz * 0.6)]
df_valid = df[int(sz * 0.6):int(sz * 0.8)]
df_test = df[int(sz * 0.8):]


print(df_train.shape)
print(df_valid.shape)
print(df_test.shape)
print(df_train)

#scaler = StandardScaler()
scaler = MaxAbsScaler()
scaler.fit(df_train)
df_train = scaler.transform(df_train)
df_valid = scaler.transform(df_valid)
df_test = scaler.transform(df_test)

num_hr = 24 * 6 * 7 + args.input_length + args.horizon

train = np.zeros((df_train.shape[0] - num_hr  + 1, df_train.shape[1] * num_hr))
valid = np.zeros((df_valid.shape[0] - num_hr + 1, df_valid.shape[1] * num_hr))
test = np.zeros((df_test.shape[0] - num_hr + 1, df_test.shape[1] * num_hr))

for i in range(train.shape[0]):
  for j in range(num_hr):
    train[i][df_train.shape[1] * j: df_train.shape[1] * (j + 1)] = df_train[i + j]

for i in range(valid.shape[0]):
  for j in range(num_hr):
    valid[i][df_valid.shape[1] * j: df_valid.shape[1] * (j + 1)] = df_valid[i + j]

for i in range(test.shape[0]):
  for j in range(num_hr):
    test[i][df_test.shape[1] * j: df_test.shape[1] * (j + 1)] = df_test[i + j]

print(train[-1].shape)
print(valid[-1].shape)
print(test[-1].shape)
print('===============')
print(test[-1])

memory_X_train_tmp = train[:, : df_train.shape[1] * 24 * 7 * 6]
memory_X_valid_tmp = valid[:, : df_valid.shape[1] * 24 * 7 * 6]
memory_X_test_tmp = test[:, : df_test.shape[1] * 24 * 7 * 6]

memory_X_train = np.zeros((memory_X_train_tmp.shape[0], df_train.shape[1] * args.input_length * 7))
memory_X_valid = np.zeros((memory_X_valid_tmp.shape[0], df_valid.shape[1] * args.input_length * 7))
memory_X_test = np.zeros((memory_X_test_tmp.shape[0], df_test.shape[1] * args.input_length * 7))

for i in range(7):
  memory_X_train[:, feature_sz * args.input_length * i: feature_sz * args.input_length * (i + 1)] = memory_X_train_tmp[:, feature_sz * 24 * 6 * i: feature_sz * 24 * 6 * i + feature_sz * args.input_length]
  memory_X_valid[:, feature_sz * args.input_length * i: feature_sz * args.input_length * (i + 1)] = memory_X_valid_tmp[:, feature_sz * 24 * 6 * i: feature_sz * 24 * 6 * i + feature_sz * args.input_length]
  memory_X_test[:, feature_sz * args.input_length * i: feature_sz * args.input_length * (i + 1)] = memory_X_test_tmp[:, feature_sz * 24 * 6 * i: feature_sz * 24 * 6 * i + feature_sz * args.input_length]

input_X_train = train[:, feature_sz * 24 * 6 * 7: feature_sz * 24 * 6 * 7 + feature_sz * args.input_length]
input_X_valid = valid[:, feature_sz * 24 * 6 * 7: feature_sz * 24 * 6 * 7 + feature_sz * args.input_length]
input_X_test = test[:, feature_sz * 24 * 6 * 7: feature_sz * 24 * 6 * 7 + feature_sz * args.input_length]

y_train = train[:, -feature_sz:]
y_valid = valid[:, -feature_sz:]
y_test = test[:, -feature_sz:]


#y_train = np.zeros((train.shape[0], 10))
#y_valid = np.zeros((valid.shape[0], 10))
#y_test = np.zeros((test.shape[0], 10))

#for i in range(train.shape[0]):
#  for j in range(1, 10, 1):
#    y_train[i, j - 1] = train[i, df_train.shape[1] * (24 * 8 + j) - 1]

#for i in range(valid.shape[0]):
#  for j in range(1, 10, 1):
#    y_valid[i, j - 1] = valid[i, df_valid.shape[1] * (24 * 8 + j) - 1]

#for i in range(test.shape[0]):
#  for j in range(1, 10, 1):
#    y_test[i, j - 1] = test[i, df_test.shape[1] * (24 * 8 + j) - 1]

memory_X_train = np.reshape(memory_X_train, (memory_X_train.shape[0], 7, args.input_length, int(memory_X_train.shape[1] // (7 * args.input_length)), 1))
memory_X_valid = np.reshape(memory_X_valid, (memory_X_valid.shape[0], 7, args.input_length, int(memory_X_valid.shape[1] // (7 * args.input_length)), 1))
memory_X_test = np.reshape(memory_X_test, (memory_X_test.shape[0], 7, args.input_length, int(memory_X_test.shape[1] // (7 * args.input_length)), 1))
input_X_train = np.reshape(input_X_train, (input_X_train.shape[0], 1, args.input_length, int(input_X_train.shape[1] // args.input_length), 1))
input_X_valid = np.reshape(input_X_valid, (input_X_valid.shape[0], 1, args.input_length, int(input_X_valid.shape[1] // args.input_length), 1))
input_X_test = np.reshape(input_X_test, (input_X_test.shape[0], 1, args.input_length, int(input_X_test.shape[1] // args.input_length), 1))
y_train = np.reshape(y_train, (y_train.shape[0], feature_sz))
y_valid = np.reshape(y_valid, (y_valid.shape[0], feature_sz))
y_test = np.reshape(y_test, (y_test.shape[0], feature_sz))
print(memory_X_train.shape)
print(memory_X_valid.shape)
print(memory_X_test.shape)
print(input_X_train.shape)
print(input_X_valid.shape)
print(input_X_test.shape)
print(y_train.shape)
print(y_valid.shape)


memory = Input((7, args.input_length, memory_X_train.shape[3], 1))
question = Input((1, args.input_length, input_X_train.shape[3], 1))

input_encoded_m = TimeDistributed(Conv2D(args.CNN_unit, kernel_size = (args.CNN_kernel, feature_sz), padding = 'valid', activation = 'relu'))(memory)
print(input_encoded_m)
input_encoded_m = Reshape((7, args.input_length - args.CNN_kernel + 1, args.CNN_unit))(input_encoded_m)
print(input_encoded_m)
input_encoded_m = attention_3d_block(input_encoded_m, args.input_length - args.CNN_kernel + 1)
input_encoded_m = TimeDistributed(GRU(args.GRU_unit, activation = 'relu', dropout = 0.2, recurrent_dropout = 0.2, kernel_initializer = 'glorot_normal', return_sequences = False))(input_encoded_m)
print(input_encoded_m)
#input_encoded_m = Reshape((7, (args.input_length - args.CNN_kernel + 1) * args.GRU_unit))(input_encoded_m)
input_encoded_m = Reshape((7, args.GRU_unit))(input_encoded_m)
print(input_encoded_m)

question_encoded = TimeDistributed(Conv2D(args.CNN_unit, kernel_size = (args.CNN_kernel, feature_sz), padding = 'valid', activation = 'relu'))(question)
question_encoded = Reshape((1, args.input_length - args.CNN_kernel + 1, args.CNN_unit))(question_encoded)
print(question_encoded)
question_encoded = attention_3d_block(question_encoded, args.input_length - args.CNN_kernel + 1)
question_encoded = TimeDistributed(GRU(args.GRU_unit, activation = 'relu', dropout = 0.2, recurrent_dropout = 0.2, kernel_initializer = 'glorot_normal', return_sequences = False))(question_encoded)
print(question_encoded)
#question_encoded = Reshape((1, (args.input_length - args.CNN_kernel + 1) * args.GRU_unit))(question_encoded)
question_encoded = Reshape((1, args.GRU_unit))(question_encoded)
print(question_encoded)

match = dot([input_encoded_m, question_encoded], axes = (2, 2))
print(match)
match = Activation('softmax')(match)
print(match)

input_encoded_c = TimeDistributed(Conv2D(args.CNN_unit, kernel_size = (args.CNN_kernel, feature_sz), padding = 'valid', activation = 'relu'))(memory)
input_encoded_c = Reshape((7, args.input_length - args.CNN_kernel + 1, args.CNN_unit))(input_encoded_c)
print(input_encoded_c)
input_encoded_c = attention_3d_block(input_encoded_c, args.input_length - args.CNN_kernel + 1)
input_encoded_c = TimeDistributed(GRU(args.GRU_unit, activation = 'relu', dropout = 0.2, recurrent_dropout = 0.2, kernel_initializer = 'glorot_normal', return_sequences = False))(input_encoded_c)
print(input_encoded_c)
#input_encoded_c = Reshape((7, (args.input_length - args.CNN_kernel + 1) * args.GRU_unit))(input_encoded_c)
input_encoded_c = Reshape((7, args.GRU_unit))(input_encoded_c)
print(input_encoded_c)

o = multiply([match, input_encoded_c])
print(o)

#question_encoded = Reshape((32 *22, ))(question_encoded)
#question_encoded = RepeatVector(7)(question_encoded)
question_encoded = Flatten()(question_encoded)
o = Flatten()(o)
#o = Reshape((640 * 6,))(o)
#print(question_encoded)
#a_hat = concatenate([question_encoded, o], axis = 2)
#a_hat = Flatten()(a_hat)
a_hat = concatenate([question_encoded, o])
#a_hat = BatchNormalization()(a_hat)
print(a_hat)

#x = Dense(1024)(a_hat)
#x = LeakyReLU()(x)
#x = Dropout(0.2)(x)
#x = Dense(512)(a_hat)
#x = LeakyReLU()(x)
#x = Dropout(0.2)(x)
#x = Dense(256)(x)
#x = LeakyReLU()(x)
#x = Dropout(0.2)(x)
#x = Flatten()(x)
#x = Dense(256)(x)
#x = LeakyReLU()(x)
#x = Dense(128)(x)
#x = LeakyReLU()(x)
#x = Dropout(0.2)(x)
main_output = Dense(y_train.shape[1])(a_hat)
print(main_output)

if args.AR:
  question_AR = Reshape((args.input_length, input_X_train.shape[3]))(question)
  print(question_AR)
  question_AR = Permute((2, 1))(question_AR)
  print(question_AR)
  #question_AR = Reshape((-1, args.input_length))(question_AR)
  #print(question_AR)
  #question_AR =TimeDistributed(Dense(1, kernel_regularizer=regularizers.l2(0.01)))(question_AR)
  question_AR =TimeDistributed(Dense(1))(question_AR)
  print(question_AR)
  question_AR = Flatten()(question_AR)
  main_output = add([main_output, question_AR])
  #main_output = Dense(y_train.shape[1])(main_output)
  #question_AR = Reshape(input_X_train.shape[2])(question_AR)
  #question_AR = Flatten()(question_AR)
  print(main_output)


#y_valid_ = y_valid * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
#y_test_ = y_test * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
y_valid_ = scaler.inverse_transform(y_valid)
y_test_ = scaler.inverse_transform(y_test)

opt = optimizers.RMSprop(lr = 0.0001, decay = 1e-8)
model = Model(inputs = [memory, question], outputs = [main_output])
model.compile(loss = 'mae', optimizer = opt, metrics = ['mse', 'mae', 'mape', RRSE])
filepath = './model/memdnn_AR_inputlength_' + str(args.input_length) + '_cnn_' + str(args.CNN_unit) + '_cnnkernel_' + str(args.CNN_kernel) + '_gru_' + str(args.GRU_unit) + '_horizon_' + str(args.horizon) + '.h5'
checkpoint = ModelCheckpoint(filepath, monitor = 'val_loss', save_weights_only = False, save_best_only = True)
earlystop = EarlyStopping(monitor = 'val_loss', patience = 100)
callbacks_list = [checkpoint, earlystop]
print(model.summary())
model.fit([memory_X_train, input_X_train], [y_train], epochs = 200, batch_size = 128, validation_data = ([memory_X_test, input_X_test], [y_test]), callbacks = callbacks_list)
model = load_model(filepath, custom_objects={'RRSE': RRSE})
predictions = scaler.inverse_transform(model.predict([memory_X_test, input_X_test], batch_size = 128))
print("testing RRSE is", rrse(y_test_.T, predictions.T))
print("testing CORR is", CORR(y_test_.T, predictions.T))
print(y_test_.shape)
print(predictions.shape)

f = open("AR_12_result.csv", "a+")
f.write(str(args.horizon) + "," + str(args.input_length) + "," + str(args.CNN_kernel) + "," + str(args.CNN_unit) + "," + str(args.GRU_unit) + "," + str(rrse(y_test_.T, predictions.T)) + "," + str(CORR(y_test_.T, predictions.T)) + "\n")

"""
best_rmse = 100
best_mae = 100

for _ in range(1000):
  print('epochs: ', _)
  model.fit([memory_X_train, input_X_train], [y_train], epochs = 1, batch_size = 128, validation_data = ([memory_X_valid, input_X_valid], [y_valid]))
  model.save(filepath)
  model = load_model(filepath)
  predictions = (model.predict([memory_X_test, input_X_test])) * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
  print("testing rmse is", rmse(y_test_, predictions))
  print("testing mae is", mae(y_test_, predictions))
  print("testing mape is", mape(y_test_, predictions))
  if best_rmse > rmse(y_test_, predictions):
    best_rmse = rmse(y_test_, predictions)
  if best_mae > mae(y_test_, predictions):
    best_mae = mae(y_test_, predictions)
  print("best testing rmse is", best_rmse)
  print("best testing mae is", best_mae)

#model = load_model(filepath)
#hist = model.fit([memory_X_train, input_X_train], [y_train], epochs = 100000, batch_size = 64, callbacks = callbacks_list, validation_data = ([memory_X_valid, input_X_valid], [y_valid]))

model = load_model(filepath)
predictions = model.predict([memory_X_valid, input_X_valid])
predictions = predictions * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
y_valid_ = y_valid * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
print(predictions)
print(y_valid_)
print("validation rmse is", rmse(y_valid_, predictions))
print("validation mse is",  mae(y_valid_, predictions))
print("validation mape is", mape(y_valid_, predictions))

predictions = model.predict([memory_X_test, input_X_test])
predictions = predictions * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
y_test_ = y_test * math.sqrt(scaler.var_[-1]) + scaler.mean_[-1]
print(predictions)
print(y_test_)
print("testing rmse is", rmse(y_test_, predictions))
print("testing mae is", mae(y_test_, predictions))
print("testing mape is", mape(y_test_, predictions))

"""
#print(np.sqrt(np.mean((predictions - y_valid_)**2)))

#with open('./train_history', 'wb') as file_pi:
#  pickle.dump(hist.history, file_pi)
