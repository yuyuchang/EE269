import numpy as np
import scipy.stats
import math
import keras
import keras.backend as K
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

def rmse(target, prediction):
  return np.sqrt(np.mean((target - prediction) ** 2))

def mae(target, prediction):
  return np.mean(np.abs(target - prediction))

def mape(target, prediction):
  return np.mean(np.abs((target - prediction) / target))

def rrse(y_true, y_pred):
  return np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def RRSE(y_true, y_pred):
  return 1000 * K.sqrt(K.sum(K.pow(y_true - y_pred, 2)) / K.sum(K.pow(y_true - K.mean(y_true), 2)))
  

def CORR(y_true, y_pred):
  N = y_true.shape[0]
  total = 0.0
  for i in range(N):
    if math.isnan(scipy.stats.pearsonr(y_true[i], y_pred[i])[0]):
      N -= 1
    else:
      total += scipy.stats.pearsonr(y_true[i], y_pred[i])[0]
  return total / N
  #return (((y_true - y_true_mean) * (y_pred - y_pred_mean)).mean(axis=  1) / (np.std(y_true, axis = 1) * np.std(y_pred, axis = 1))).mean()
