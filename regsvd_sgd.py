import numpy as np
import pandas as pd
import datetime as dt
import math
from evaluation import *
import numexpr as ne

def assign_id(data):
    
    items = pd.unique(data['itemid'])
    vks = pd.unique(data['vk'])
    
    itemid = {items[i]:i for i in range(items.shape[0])}
    vkid = {vks[i]:i for i in range(vks.shape[0])}

    return itemid, vkid

def split_data(data, split_time):

    training_data = data.loc[data['real_time'] <= split_time]
    test_data = data.loc[data['real_time'] > split_time]

    training_data.index = range(training_data.shape[0])
    test_data.index = range(test_data.shape[0])

    return training_data, test_data

def user_bias(data, itemid, vkid):

    avg = np.mean(data['action'])

    vk_avg = np.zeros(len(vkid))
    vk_count = np.zeros(len(vkid))

    item_avg = np.zeros(len(itemid))
    item_count = np.zeros(len(itemid))

    #update the bias for user and item
    for idx, value in data[['vk','itemid','action']].iterrows():
	
	vk_avg[vkid[value['vk']]] = vk_avg[vkid[value['vk']]] + value['action']
	vk_count[vkid[value['vk']]] = vk_count[vkid[value['vk']]] + 1

	item_avg[itemid[value['itemid']]] = item_avg[itemid[value['itemid']]] + value['action']
	item_count[itemid[value['itemid']]] = item_count[itemid[value['itemid']]] + 1 
    
    item_avg = item_avg * 1.0 / item_count - 0.5 * avg
    vk_avg = vk_avg * 1.0 / vk_count - 0.5 * avg

    return avg, item_avg, vk_avg

def sgd(data, avg, vk_avg, item_avg, u, v, itemid, vkid, gamma, lda):

    """The stochastic gradient descent for Matrix factorization"""

    sum_err = 0 

    for idx, value in data[['vk','itemid','action']].iterrows():
	
	vk_num = vkid[value['vk']]
	item_num = itemid[value['itemid']]

#	err = value['action'] - avg - vk_avg[vk_num] - item_avg[item_num] - np.dot(u[vk_num, :], v[item_num, :]) 
	err = value['action'] - np.dot(u[vk_num, :], v[item_num, :]) 

#	if np.isnan(np.dot(u[vk_num,:], v[item_num,:])):
#	    exit()

	vt = v[item_num, :]
	ut = u[vk_num, :]

	u[vk_num, :] = ut + gamma * (err * vt - lda * ut)
	v[item_num, :] = vt + gamma * (err * ut - lda * vt)
	sum_err += (err ** 2)

    sum_err = math.sqrt(sum_err / data.shape[0])
    return u, v, sum_err 

def mf(data, split_time, k):

    training_data, test_data = split_data(data, split_time)
    
    tr_itemid, tr_vkid = assign_id(training_data)
    tst_itemid, tst_vkid = assign_id(test_data)

    #initialize the bias and offect
    avg, item_avg, vk_avg = user_bias(training_data, tr_itemid, tr_vkid)
   
    u = np.random.rand(len(tr_vkid), k)
    v = np.random.rand(len(tr_itemid), k)
   
    #step size and reguarlization 
    gamma = 0.05
    lda = 0.1
    beta = 0.99
    old_err = 1e10 
    err = 1e5 

#    while(old_err - err > 0.001):
#
#	old_err = err
#	u, v, err = sgd(training_data, avg, vk_avg, item_avg, u, v, tr_itemid, tr_vkid, gamma, lda)
#	gamma = gamma * beta	
#	print '----------Error------------'
#	print err
  
    rmse, avg_rmse, new_vk, new_item = rmse_test(test_data, u, v, tr_itemid, tr_vkid, avg, item_avg, vk_avg)

    print 'MF RMSE:', rmse
    print 'AVG RMSE:', avg_rmse
    print 'Number of New user:%d, Num of total user:%d' % (new_vk, len(tst_vkid))
    print 'Number of New Items:%d, Num of total item:%d' % (new_item, len(tst_itemid))
