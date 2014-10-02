import pandas as pd
import numpy as np
import datetime as dt

def num_new(data):

    time_range = np.sort(data['real_time'].unique())
    
    log = open('log','w')

    for t in time_range[2:len(time_range):1]:

	tr_data = data.loc[data['real_time'] <= t]
	tst_data = data.loc[np.logical_and((data['real_time'] <= (t + np.timedelta64(1, 'h'))), (data['real_time'] > t)) ]

	tr_vk = pd.unique(tr_data['vk'])
	tr_item = pd.unique(tr_data['itemid'])

	tst_vk = pd.unique(tst_data['vk'])
	tst_item = pd.unique(tst_data['itemid'])

	tr_vk = set(tr_vk.flat)
	tst_vk = set(tst_vk.flat)

	tr_item = set(tr_item.flat)
	tst_item = set(tst_item.flat)

	total_user = len(tst_vk)#.shape[0]
	total_item = len(tst_item)
	new_vk = 0
	new_item = 0

	for vk in tst_vk:
	    if vk not in tr_vk:
		new_vk += 1
    
	for item in tst_item:
	    if item not in tr_item:
		new_item += 1

	print str(t) + ' new user:%d, total user:%d, fraction: %f' % (new_vk, total_user, new_vk * 100.0 / total_user)
	print str(t) + ' new item:%d, total item:%d, fraction: %f' % (new_item, total_item, new_item * 100.0 / total_item)
	print '---------------------------------'
	
	log.write(str(t) + ' new user:%d, total user:%d, fraction: %f\n' % (new_vk, total_user, new_vk * 100.0 / total_user))
	log.write(str(t) + ' new item:%d, total item:%d, fraction: %f' % (new_item, total_item, new_item * 100.0 / total_item))
	log.write('---------------------------------')
