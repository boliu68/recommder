import pandas as pd
import numpy as np
import datetime as dt

def sum_unique(x):
    return np.unique(x).shape[0]

def analyze_count(data):
    
    """real time, vk, itemid, action"""

    dsct_vk = pd.unique(data['vk'])
    dsct_itemid = pd.unique(data['itemid'])

    print 'number of user:', dsct_vk.shape
    print 'number of items:', dsct_itemid.shape
    print 'the number of ratings:', data.shape

    print 'unique actions:', pd.unique(data['action'])
    print 'the number of action 0:', np.sum(data['action'] == 0)
    print 'the number of action 1:', np.sum(data['action'] == 1)
    print 'the number of action 2:', np.sum(data['action'] == 2)
    print 'the number of action 3:', np.sum(data['action'] == 3)
    print 'the number of action 4:', np.sum(data['action'] == 4)
    
    time_range_item = data.groupby('itemid')['real_time'].aggregate(sum_unique)
    print 'Max Range:', np.max(time_range_item)
    print 'Mean Range:', np.mean(time_range_item)
    print 'Median Range:', np.median(time_range_item)
