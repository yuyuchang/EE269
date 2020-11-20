import numpy as np
import pandas as pd
from glob import glob
import datetime

def tomorrow(date):
    date = datetime.date(*list(map(int, date.split('-'))))
    date = date + datetime.timedelta(days=1)
    return '{:04d}-{:02d}-{:02d}'.format(date.year, date.month, date.day)

def rmse(target, prediction):
  return np.sqrt(np.nanmean((target - prediction) ** 2))

def mae(target, prediction):
  return np.nanmean(np.abs(target - prediction))

def mape(target, prediction):
  return np.nanmean(np.abs((target - prediction) / target))


def smape(y_true, y_pred):
  diff = np.abs(y_pred - y_true)
  factor = (np.abs(y_pred) + np.abs(y_true))
  y = diff / factor
  #y[np.isnan(y)] = 0
  return 200 * np.nanmean(y)

def evaluate_internal(sub, month=3):
    sub = pd.read_csv(sub)
    # sub.columns = ['submit_date', 'test_id', 'PM2.5', 'PM10', 'O3']
    sub.sort_values(by=['submit_date', 'test_id'], inplace=True)
    print(sub.head())
    if month == 3:
        ans = pd.read_csv('./data/internal_answer.csv')
        start_date = '2018-02-28'
    elif month == 4:
        ans = pd.read_csv('./data/internal_answer_4.csv')
        start_date = '2018-03-31'
    else:
        ans = pd.read_csv('../../data/kdd2018/answer_5.csv')
        start_date = '2018-05-01'

    days = sub.shape[0]//(48**2)
    columns = ['PM2.5', 'PM10', 'O3']

    ans.sort_values(by=['submit_date', 'test_id'], inplace=True)
    ans =ans[columns]
    sub = sub[columns]

    sub[sub < 0] = np.nan
    sub.fillna(0, inplace=True)

    ans[ans < 0] = np.nan
    #ans.loc[ans.isnull().any(axis=1), :] = np.nan

    sz = 2304
    smape_scores = []
    mae_scores  = []
    rmse_scores = []
    mape_scores = []
    x = start_date
    for i in range(days):

        smape_scores.append(smape(ans.values[i*sz:i*sz+sz,:], sub.values[i*sz:i*sz+sz,:]))
        mae_scores.append(mae(ans.values[i*sz:i*sz+sz,:], sub.values[i*sz:i*sz+sz,:]))
        rmse_scores.append(rmse(ans.values[i*sz:i*sz+sz,:], sub.values[i*sz:i*sz+sz,:]))
        mape_scores.append(mape(ans.values[i*sz:i*sz+sz,:], sub.values[i*sz:i*sz+sz,:]))
        x = tomorrow(x)

    print('smape', np.mean(smape_scores))
    print('mae', np.mean(mae_scores))
    print('rmse', np.mean(rmse_scores))
    print('mape', np.mean(mape_scores))

if __name__ == '__main__':
    files = glob('final.csv')
    evaluate_internal('lightgbm/final.csv', 5)
    # files = glob('../../result/kdd2018/*')
    for f in files:
        print(f)
        evaluate_internal(f, 5)
