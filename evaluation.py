import pandas as pd
import numpy as np
import math
from multiprocessing import Pool

def rmse_test(test_data, u, v, itemid, vk_id, avg, item_avg, vk_avg):

    """Calculate RMSE for test data
    the test data split by time"""

    rmse = 0
    rmse_count = 0

    avg_rmse = 0

    for idx, value in test_data[['vk', 'itemid', 'action']].iterrows():

	if value['vk'] not in vk_id.keys():
	    continue
	if value['itemid'] not in itemid.keys():
	    continue
	
	#rmse += (value['action'] - (np.dot(u[vk_id[value['vk']]], v[itemid[value['itemid']]]) + avg + item_avg[itemid[value['itemid']]] + vk_avg[vk_id[value['vk']]])) ** 2
	rmse += (value['action'] - (np.dot(u[vk_id[value['vk']]], v[itemid[value['itemid']]]))) ** 2
	avg_rmse += (value['action'] - (avg + item_avg[itemid[value['itemid']]] + vk_avg[vk_id[value['vk']]])) ** 2
	rmse_count += 1
    
    rmse = math.sqrt(rmse * 1.0 / rmse_count)
    avg_rmse = math.sqrt(avg_rmse * 1.0 / rmse_count)

    new_vk = 0
    new_item = 0

    for vks in pd.unique(test_data['vk']):
	if vks not in vk_id.keys():
	    new_vk += 1

    for items in pd.unique(test_data['itemid']):
	if items not in itemid.keys():
	    new_item += 1

    return rmse, avg_rmse, new_vk, new_item
