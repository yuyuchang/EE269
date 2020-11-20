import numpy as np 

epilson = 1e-8
citymeta = {
    'beijing': {
        'schema': {
            'time': ['quarter_cos', 'quarter_sin', 'month_cos', 'month_sin', 
                'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin'],
            'area': ['other',  'suburban', 'traffic', 'urban'],
            'station': [
                'dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq',
                'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq', 'zhiwuyuan_aq',
                'fengtaihuayuan_aq', 'yungang_aq', 'gucheng_aq', 'fangshan_aq',
                'daxing_aq', 'yizhuang_aq', 'tongzhou_aq', 'shunyi_aq',
                'pingchang_aq', 'mentougou_aq', 'pinggu_aq', 'huairou_aq',
                'miyun_aq', 'yanqin_aq', 'dingling_aq', 'badaling_aq',
                'miyunshuiku_aq', 'donggaocun_aq', 'yongledian_aq', 'yufa_aq',
                'liulihe_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
                'aotizhongxin_aq', 'nansanhuan_aq', 'dongsihuan_aq'],
            'aq_input':  ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2'],
            'aq_output': ['PM2.5', 'PM10', 'O3'],

        },
        'station2area': {
            'dongsi_aq': 'urban',
            'tiantan_aq': 'urban',
            'guanyuan_aq':'urban',
            'wanshouxigong_aq':'urban',
            'aotizhongxin_aq':'urban',
            'nongzhanguan_aq':'urban',
            'wanliu_aq':'urban',
            'beibuxinqu_aq':'urban',
            'zhiwuyuan_aq':'urban',
            'fengtaihuayuan_aq':'urban',
            'yungang_aq':'urban',
            'gucheng_aq':'urban',
            'fangshan_aq':'suburban',
            'daxing_aq':'suburban',
            'yizhuang_aq':'suburban',
            'tongzhou_aq':'suburban',
            'shunyi_aq':'suburban',
            'pingchang_aq':'suburban',
            'mentougou_aq':'suburban',
            'pinggu_aq':'suburban',
            'huairou_aq':'suburban',
            'miyun_aq':'suburban',
            'yanqin_aq':'suburban',
            'dingling_aq':'other',
            'badaling_aq':'other',
            'miyunshuiku_aq':'other',
            'donggaocun_aq':'other',
            'yongledian_aq':'other',
            'yufa_aq':'other',
            'liulihe_aq':'other',
            'qianmen_aq':'traffic',
            'yongdingmennei_aq': 'traffic',
            'xizhimenbei_aq': 'traffic',
            'nansanhuan_aq': 'traffic',
            'dongsihuan_aq': 'traffic'
        },
        'stationIds': [
            'dongsi_aq', 'tiantan_aq', 'guanyuan_aq', 'wanshouxigong_aq',
            'nongzhanguan_aq', 'wanliu_aq', 'beibuxinqu_aq', 'zhiwuyuan_aq',
            'fengtaihuayuan_aq', 'yungang_aq', 'gucheng_aq', 'fangshan_aq',
            'daxing_aq', 'yizhuang_aq', 'tongzhou_aq', 'shunyi_aq',
            'pingchang_aq', 'mentougou_aq', 'pinggu_aq', 'huairou_aq',
            'miyun_aq', 'yanqin_aq', 'dingling_aq', 'badaling_aq',
            'miyunshuiku_aq', 'donggaocun_aq', 'yongledian_aq', 'yufa_aq',
            'liulihe_aq', 'qianmen_aq', 'yongdingmennei_aq', 'xizhimenbei_aq',
            'aotizhongxin_aq', 'nansanhuan_aq', 'dongsihuan_aq'],
        'statistic': {
            'mu': np.asarray([60.042, 90.86,  59.554, 45.603,  0.945,  9.079]),
            'std': np.asarray([67.699, 89.078, 55.399, 31.727,  0.952, 12.586])
        }
    },
    'london': {
        'schema': {
            'time': ['quarter_cos', 'quarter_sin', 'month_cos', 'month_sin', 
                'weekday_cos', 'weekday_sin', 'hour_cos', 'hour_sin'],
            'area': ['Suburban', 'Urban', 'Roadside', 'Kerbside', 'Industrial'],
            'station': ['BL0', 'CD9','CD1', 'GN0', 'GR4', 'GN3',
                'GR9', 'HV1', 'KF1', 'LW2', 'ST5', 'TH4', 'MY7'],
            'aq_input':  ['PM2.5', 'PM10', 'O3', 'NO2', 'CO', 'SO2', 'NO', 'NOX'],
            'aq_output': ['PM2.5', 'PM10'],

        },
        'station2area': {
            'BX9':'Suburban',
            'BX1':'Suburban',
            'BL0':'Urban',
            'CD9':'Roadside',
            'CD1':'Kerbside',
            'CT2':'Kerbside',
            'CT3':'Urban',
            'CR8':'Urban',
            'GN0':'Roadside',
            'GR4':'Suburban',
            'GN3':'Roadside',
            'GR9':'Roadside',
            'GB0':'Roadside',
            'HR1':'Urban',
            'HV1':'Roadside',
            'LH0':'Urban',
            'KC1':'Urban',
            'KF1':'Urban',
            'LW2':'Roadside',
            'RB7':'Urban',
            'TD5':'Suburban',
            'ST5':'Industrial',
            'TH4':'Roadside',
            'MY7':'Kerbside'
        },
        'stationIds': ['BL0', 'CD9','CD1', 'GN0', 'GR4', 'GN3',
            'GR9', 'HV1', 'KF1', 'LW2', 'ST5', 'TH4', 'MY7'
        ],
        'statistic': {
            'mu': np.asarray([13.438, 21.073, 34.844, 43.167, 0.285, 2.378, 38.406, 101.994]),
            'std': np.asarray([10.714, 13.988, 14.715, 26.651, 0.054, 1.067, 57.323, 110.288])
        }
    }
}
