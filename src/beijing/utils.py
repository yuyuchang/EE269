import numpy as np

def rmse(target, prediction):
  return np.sqrt(np.mean((target - prediction) ** 2))

def mae(target, prediction):
  return np.mean(np.abs(target - prediction))

def mape(target, prediction):
  return np.mean(np.abs((target - prediction) / target))

def rrse(y_true, y_pred):
  return np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def CORR(y_true, y_pred):
  y_true_mean = np.reshape(np.mean(y_true, axis = 1), (-1, 1))
  y_pred_mean = np.reshape(np.mean(y_pred, axis = 1), (-1, 1))
  y_true_std = np.std(y_true, axis = 1)
  y_pred_std = np.std(y_pred, axis = 1)
  return (((y_true - y_true_mean) * (y_pred - y_pred_mean)).mean(axis=  1) / (np.std(y_true, axis = 1) * np.std(y_pred, axis = 1))).mean()
