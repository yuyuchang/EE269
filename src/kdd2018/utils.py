# -*- coding: utf-8 -
import numpy as np 
import pandas as pd
import pickle
from random import random, shuffle
import datetime
import string
import math
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor
from constant import *

def load_pkl(path):
	return pickle.load(open( path, 'rb'))	

def dump_pkl(file, path):
	pickle.dump(file, open( path, 'wb'))	

def load_npy(path):
	file = np.load(path)	
	return file 

def dump_npy(file, path):
	np.save(path, file)	

def load_txt(path):
	return np.genfromtxt(path, dtype='str')

def dump_txt(file, path):
	np.savetxt(path, file, delimiter=',')

def load_meta(meta_dir):
    en2ch = load_pkl('{}/en2ch.pkl'.format(meta_dir))
    ch2en = load_pkl('{}/ch2en.pkl'.format(meta_dir))
    station_info = load_pkl('{}/station_info.pkl'.format(meta_dir))
    areas = load_pkl('{}/areas.pkl'.format(meta_dir))
    return en2ch, ch2en, station_info, areas
    pass

def load_data(args, meta, flag):
    df = pd.read_csv('{}/{}_{}.csv'.format(args.data_dir, args.city, flag))
    stationIds = meta['stationIds']
    station2area = meta['station2area']
    df = df.fillna(0)
    df = df.loc[df['stationId'].isin(stationIds)]
    df = df.sort_values(by=['stationId', 'date'])
    aq_input = meta['schema']['aq_input']
    mu = meta['statistic']['mu']
    std = meta['statistic']['std']
    df[aq_input] = (df[aq_input] - mu) / std
    df['area'] = df['stationId'].apply(lambda x: station2area[x])
    df = parse_time(df)
    area = pd.get_dummies(df['area'])
    station = pd.get_dummies(df['stationId'])
    df = df.join(area)
    df = df.join(station)
    df = tranform_corordinate(df)
    data = {}
    areas = group_area(df)
    for a, a_df in areas.items():
        data[a] = group_station(a_df, meta['schema'])
    return data

def group_area(df):
    areas = {}
    df = df.groupby('area')
    for g ,data in df:
        areas[g] = data
    return areas

def group_station(df, schema):
    stations = {}
    df = df.groupby('stationId')
    for g ,data in df:
         stations[g] = {}
         input_schema = schema['aq_input'] + schema['time'] + schema['area'] + schema['station']
         stations[g]['input'] = data[input_schema].values
         output_schema = schema['aq_output']
         stations[g]['output'] = data[output_schema].values
    return stations

def to_polor(col, period):
    col1 = col.apply(lambda x: math.cos(2*math.pi *x /period))
    col2 = col.apply(lambda x: math.sin(2*math.pi *x /period))
    return col1, col2

def parse_time(df):
    df['utc_time'] = pd.to_datetime(df['utc_time'])
    df['year'] = df['utc_time'].dt.year
    df['quarter'] = df['utc_time'].dt.quarter
    df['month'] = df['utc_time'].dt.month
    df['week'] = df['utc_time'].dt.week
    df['weekday'] = df['utc_time'].apply(lambda x: x.weekday())
    df['date'] = df['utc_time'].dt.date
    df['day'] = df['utc_time'].dt.day
    df['hour'] = df['utc_time'].dt.hour
    return df   

def tranform_corordinate(df):
    df['quarter_cos'], df['quarter_sin'] = to_polor(df['quarter'], 4)
    df['month_cos'], df['month_sin'] = to_polor(df['month'], 31)
    df['weekday_cos'], df['weekday_sin'] = to_polor(df['weekday'], 7)
    df['hour_cos'], df['hour_sin'] = to_polor(df['hour'], 24)
    return df
   
def split_data(data, r):
    valid = int(len(data['input']) * (1 - r))
    data_tr = {}
    data_val = {}
    data_tr['input'], data_val['input'] = data['input'][:valid], data['input'][valid:]
    data_tr['output'], data_val['output'] = data['output'][:valid], data['output'][valid:]
    return data_tr, data_val

def split_data_neighbor(data, r):
    data_tr = {}
    data_val = {}
    for k in data.keys():
        valid = int(len(data[k]['input']) * (1 - r))
        data_tr[k] = {}
        data_val[k] = {}
        data_tr[k]['input'], data_val[k]['input'] = data[k]['input'][:valid], data[k]['input'][valid:]
        data_tr[k]['output'], data_val[k]['output'] = data[k]['output'][:valid], data[k]['output'][valid:]
        
    return data_tr, data_val

def shuffle_arrs( *arrs):
    r = random()
    result = []
    for arr in arrs:       
        shuffle(arr, lambda : r)   
        result.append(arr)
    return tuple(result)
