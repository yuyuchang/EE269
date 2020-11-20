# -*- coding: utf-8 -
import numpy as np 
import pandas as pd
import pickle
import datetime
import string
import math
from utils import *
from torch.utils.data import Dataset, DataLoader
from torch import FloatTensor

schema = ['PM2.5','PM10','O3','NO2','CO','SO2']
d_schema = ['quarter_cos', 'quarter_sin', 'month_cos', 'month_sin', 
    'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin']
a_schema = ['other',  'suburban', 'traffic', 'urban']

# class AirDataSet(Dataset):
#     def __init__(self, data = None, isTrain = True):
#         self.data = data
#         self.isTrain = isTrain

#     def __len__(self):
#         return len(self.data['input'])

#     def __getitem__(self, idx):
#         input = FloatTensor(self.data['input'][idx])
#         if(self.isTrain) :
#             output = FloatTensor(self.data['output'][idx])
#             sample = {'input': input, 'output': output}
#         else :
#             sample = {'input': input}
#         return sample

# class NeighborAirDataSet(Dataset):
#     def __init__(self, data = None, isTrain = True):
#         self.data = data
#         self.isTrain = isTrain
#     def __len__(self):
#         c = list(self.data.keys())[0]
#         return len(self.data[c]['input'])
#     def __getitem__(self, idx):
#         c = choice(list(self.data.keys()))
#         neighbors = []
#         for k in  self.data.keys():
#             if k == c:
#                 input = self.data[k]['input'][idx]
#             else :
#                 neighbors.append(self.data[k]['input'][idx])
#         input = FloatTensor(input)
#         neighbors = FloatTensor(neighbors)
#         if(self.isTrain) :
#             output = FloatTensor(self.data[c]['output'][idx])
#             sample = {'input': input, 'neighbors': neighbors, 'output': output}
#         else :
#             sample = {'input': input, 'neighbors': neighbors}
#         return sample


class AirDataSet(Dataset):
    def __init__(self, args, aq_data = None, meo_data = None):
        self.aq_data = aq_data
        self.meo_data = meo_data
        self.prev = args.prev
        self.start = args.start
        self.end = args.end
        self.isTrain = args.train
        self.flag = args.input_flag
        self.stations = list(self.aq_data.keys())
        self.numofstation = len(self.aq_data.keys())
        self.length = len(self.aq_data[self.stations[0]]['input']) - self.prev - self.end

    def __len__(self):
        return  self.length * self.numofstation
    
    def get_feat_item(self, idx):
        pass
        
    def get_oigin_item(self, c, idx):
        i = idx
        j = idx + self.prev
        input = self.aq_data[c]['input'][i:j]
        input = FloatTensor(input)
        if(self.isTrain) :
            i = idx + self.prev + self.start
            j = idx + self.prev + self.end
            output = self.aq_data[c]['output'][i:j]
            output = FloatTensor(output)
            sample = {'input': input, 'output': output}
        else :
            sample = {'input': input}
        return sample

    def get_neighbor_item(self, c, idx):
        i = idx
        j = idx + self.prev
        neighbors = []
        for k in  self.data.keys():
            if k == c:
                input = self.aq_data[k]['input'][i:j]
            else :
                neighbors.append(self.aq_data[k]['input'][i:j])
        input = FloatTensor(input)
        neighbors = FloatTensor(neighbors)
        if(self.isTrain) :
            i = idx + self.prev + self.start
            j = idx + self.prev + self.end
            output = self.aq_data[c]['output'][i:j]
            output = FloatTensor(output)
            sample = {'input': input, 'neighbors': neighbors, 'output': output}
        else :
            sample = {'input': input, 'neighbors': neighbors}
        return sample

    def __getitem__(self, idx):        
        c = idx // self.length
        c = self.stations[c]
        idx = idx % self.length
        if(self.flag == 'neighbor'):
            return self.get_neighbor_item(c, idx)
        else :
            return self.get_oigin_item(c, idx)

# def date2date_station(df, start, end):
#     df = df.groupby('stationId')
#     data = {}
#     for g1, s in df:
#         data[g1] = {}
#         s = s.groupby('date')
#         fs = []
#         hs = []
#         inputs = []
#         outputs = []
#         for g2, d in s:
#             f = d[schema + d_schema + a_schema].values
#             fs.append(f)
#             h = d[['PM2.5','PM10','O3']].values
#             hs.append(h)
#         fs = fs[:-2]
#         hs = hs[2:]
#         for i in range(len(fs) - 1):
#             f = np.concatenate((fs[i], fs[i+1]), axis=0)
#             inputs.append(f)
#             h = np.concatenate((hs[i], hs[i+1]), axis=0)
#             h =h[start:end]
#             outputs.append(h)
#         data[g1]['input'] = np.asarray(inputs)        
#         data[g1]['output'] = np.asarray(outputs)        
#     return data

# def hour2hour_station(df, start, end):
#     df = df.groupby('stationId')
#     data = {}
#     for g1, d in df:
#         data[g1] = {}
#         inputs = []
#         outputs = []
#         fs = d[schema+ d_schema + a_schema].values
#         hs = d[['PM2.5','PM10','O3']].values
#         fs = fs[:-48]
#         hs = hs[48:]
#         for i in range(len(fs) - 47):
#             f =fs[i:i+48]
#             inputs.append(f)
#             h =hs[i:i+48]
#             h =h[start:end]
#             outputs.append(h)
#         data[g1]['input'] = np.asarray(inputs)
#         data[g1]['output'] = np.asarray(outputs)
#     return data

# def date2date_area(df, start, end):
#     df = df.groupby('area')
#     data = {}
#     for g1, a in df:
#         data[g1] = {}
#         s_df = a.groupby('stationId')
#         fs = []
#         hs = []
#         inputs = []
#         outputs = []
#         for g2, t in s_df:
#             t_df = t.groupby('date')
#             for g3, d in t_df:
#                 f = d[schema+ d_schema + a_schema].values
#                 fs.append(f)
#                 h = d[['PM2.5','PM10','O3']].values
#                 hs.append(h)
#             fs = fs[:-2]
#             hs = hs[2:]
#             for i in range(len(fs) - 1):
#                 f = np.concatenate((fs[i], fs[i+1]), axis=0)
#                 inputs.append(f)
#                 h = np.concatenate((hs[i], hs[i+1]), axis=0)
#                 h = h[start:end]
#                 outputs.append(h)
#         data[g1]['input'] = np.asarray(inputs)        
#         data[g1]['output'] = np.asarray(outputs)      
#     return data

# def hour2hour_area(df, start, end):
#     df = df.groupby('area')
#     data = {}
#     for g1, a in df:
#         s_df = a.groupby('stationId')
#         inputs = []
#         outputs = []
#         data[g1] = {}
#         for g2, s in s_df:
#             fs = s[schema+ d_schema + a_schema].values
#             hs = s[['PM2.5','PM10','O3']].values
#             fs = fs[:-48]
#             hs = hs[48:]
#             for i in range(len(fs) - 47):
#                 f =fs[i:i+48]
#                 inputs.append(f)
#                 h =hs[i:i+48]
#                 h =h[start:end]
#                 outputs.append(h)
#         data[g1]['input'] = np.asarray(inputs)
#         data[g1]['output'] = np.asarray(outputs)
#     return data


# def date2date_all(df, start, end):
#     df = df.groupby('stationId')
#     data = {}
#     data['all'] = {}
#     inputs = []
#     outputs = []
#     for g1, s in df:
#         s = s.groupby('date')
#         fs = []
#         hs = []
#         for g2, d in s:
#             f = d[schema+ d_schema + a_schema].values
#             fs.append(f)
#             h = d[['PM2.5','PM10','O3']].values
#             hs.append(h)
#         fs = fs[:-2]
#         hs = hs[2:]
#         for i in range(len(fs) - 1):
#             f = np.concatenate((fs[i], fs[i+1]), axis=0)
#             inputs.append(f)
#             h = np.concatenate((hs[i], hs[i+1]), axis=0)
#             h =h[start:end]
#             outputs.append(h)
#     data['all']['input'] = np.asarray(inputs)        
#     data['all']['output'] = np.asarray(outputs)        
#     return data

# def hour2hour_all(df, start, end):
#     df = df.groupby('stationId')
#     data = {}
#     data['all'] = {}
#     inputs = []
#     outputs = []
#     for g1, d in df:
#         data[g1] = {}
#         inputs = []
#         outputs = []
#         fs = d[schema+ d_schema + a_schema].values
#         hs = d[['PM2.5','PM10','O3']].values
#         fs = fs[:-48]
#         hs = hs[48:]
#         for i in range(len(fs) - 47):
#             f =fs[i:i+48]
#             inputs.append(f)
#             h =hs[i:i+48]
#             h =h[start:end]
#             outputs.append(h)
#     data['all']['input'] = np.asarray(inputs)
#     data['all']['output'] = np.asarray(outputs)
#     return data


# def date2date_neighbor(df, start, end):
#     df = df.groupby('area')
#     data = {}
#     for g1, a in df:
#         data[g1] = {}
#         s_df = a.groupby('stationId')
#         fs = []
#         hs = []
#         inputs = []
#         outputs = []
#         for g2, t in s_df:
#             data[g1][g2] = {}
#             t_df = t.groupby('date')
#             for g3, d in t_df:
#                 f = d[schema+ d_schema + a_schema].values
#                 fs.append(f)
#                 h = d[['PM2.5','PM10','O3']].values
#                 hs.append(h)
#             fs = fs[:-2]
#             hs = hs[2:]
#             for i in range(len(fs) - 1):
#                 f = np.concatenate((fs[i], fs[i+1]), axis=0)
#                 inputs.append(f)
#                 h = np.concatenate((hs[i], hs[i+1]), axis=0)
#                 h = h[start:end]
#                 outputs.append(h)
#             data[g1][g2]['input'] = np.asarray(inputs)        
#             data[g1][g2]['output'] = np.asarray(outputs)      
#     return data

# def hour2hour_neighbor(df, start, end):
#     df = df.groupby('area')
#     data = {}
#     for g1, a in df:
#         s_df = a.groupby('stationId')
#         inputs = []
#         outputs = []
#         data[g1] = {}
#         for g2, s in s_df:
#             data[g1][g2] = {}
#             fs = s[schema+ d_schema + a_schema].values
#             hs = s[['PM2.5','PM10','O3']].values
#             fs = fs[:-48]
#             hs = hs[48:]
#             for i in range(len(fs) - 47):
#                 f =fs[i:i+48]
#                 inputs.append(f)
#                 h =hs[i:i+48]
#                 h =h[start:end]
#                 outputs.append(h)
#             data[g1][g2]['input'] = np.asarray(inputs)
#             data[g1][g2]['output'] = np.asarray(outputs)
#     return data

# def date2date(df):
#     df = df.groupby('stationId')
#     data = {}
#     for g1, s in df:
#         data[g1] = {}
#         s = s.groupby('date')
#         fs = []
#         hs = []
#         for g2, d in s:
#             f = np.mean(d[schema+ d_schema+ a_schema].values, 0)
#             fs.append(f)
#             h = np.mean(d[['PM2.5','PM10','O3']].values, 0)
#             hs.append(h)
#         fs = np.asarray(fs)
#         hs = np.asarray(hs[10:])
#         inputs = []
#         outputs = []
#         for i in range(len(hs) - 1):
#             inputs.append(fs[i:i+10])
#             outputs.append(hs[i:i+2])
#         data[g1]['input'] = np.asarray(inputs)
#         data[g1]['output'] = np.asarray(outputs)
#     return data

# def hour2date(df):
#     df = df.groupby('stationId')
#     data = {}
#     for g1, s in df:
#         data[g1] = {}
#         s = s.groupby('date')
#         fs = []
#         hs = []
#         for g2, d in s:
#             f = d[schema+ d_schema+ a_schema].values
#             fs.append(f)
#             h = np.mean(d[['PM2.5','PM10','O3']].values, 0)
#             hs.append(h)
#         fs = np.asarray(fs[:-2])
#         hs = np.asarray(hs[2:])
#         inputs = []
#         outputs = []
#         for i in range(len(fs) - 1):
#             f = np.concatenate((fs[i], fs[i+1]), axis=0)
#             inputs.append(f)
#             h = hs[i:i+2]
#             outputs.append(h)
#         data[g1]['input'] = inputs
#         data[g1]['output'] = outputs
#     return data
