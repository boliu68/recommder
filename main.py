import read_data as rd
import basic_analyze as az
import regsvd_sgd as mf
import datetime as dt

if __name__ == '__main__':

    path = '/home/bliuab/tencent/data/all'
    data = rd.read_data(path)
    #az.analyze_count(data)

    #the data before split_time is for training, after that for testing
    k = 100
    split_time = dt.datetime(2014,9,24,1)
    mf.mf(data, split_time, k)
