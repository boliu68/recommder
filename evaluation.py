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

def ndcg_test(test_data, prediction, itemid, vkid, k):

    gd = np.zeros((len(vkid), len(itemid)))
    new_user = 0
    new_item = 0

    for idx, value in test_data[['vk', 'itemid', 'action']].iterrows():

	if value['vk'] not in vkid.keys():
	    continue
	if value['itemid'] not in itemid.keys():
	    continue

	gd[vkid[value['vk']], itemid[value['itemid']]] = value['action']

    ndcg = 0

    #only for those exist entries
    n = 0

    for tst_vkid in pd.unique(test_data['vk']):
		
	if tst_vkid not in vkid.keys():
	    continue
	
	n += 1
	gd_user = gd[vkid[tst_vkid], :]
	pred_user = prediction[vkid[tst_vkid], :]
	
	sort_idx = np.argsort(pred_user)[::-1]
	gd_user_k = gd_user[sort_idx[:k]]
	sort_idx_k = sort_idx[:k] + 1
	
	print gd_user_k
	print pred_user[sort_idx_k - 1]

	dcg_k = np.sum((np.power(2, gd_user_k) - 1) / np.log2(sort_idx_k))
	
	sort_idx = np.argsort(gd_user)[::-1]
	gd_user_k = gd_user[sort_idx[::k]]
	sort_idx_k = sort_idx[::k] + 1
	normalize = np.sum((np.power(2, gd_user_k - 1)) / np.log2(sort_idx_k))
	ndcg += dcg_k / normalize
	
	print dcg_k / normalize
	print '--------------'
    
    return ndcg / n 
