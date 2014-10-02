import pandas as pd
import datetime as dt

def date_parser(x):
    return dt.datetime.strptime(x, '%Y%m%d%H')

def read_data(path):

    """read a single file from the path"""
    data = pd.read_csv(path, sep='\01', names=['real_time','vk','itemid','action', 'shopid','name'], parse_dates='real_time', date_parser=date_parser)
    #data = pd.read_csv(path, sep='\01', names=['real_time','vk','itemid','action', 'shopid','name'])
    return data

if __name__ == '__main__':

    path = '/home/bliuab/tencent/data/all'
    read_data(path)
