import pandas as pd
import numpy as np

def random_sample(data):

    """random sample a subset a data
    for both training and test"""

    vk_size = data.groupby('vk').size()
    item_size = data.groupby('itemid').size()

    vk_size.sort(ascending=False)
    item_size.sort(ascending=False)
    
    top_vk = vk_size[vk_size >= 80]
    top_item = item_size[item_size >= 800]
    
    print '-----VK------'
    print top_vk.shape
    print vk_size.shape

    print '-----Item-----'
    print top_item.shape
    print item_size.shape

    is_top_vk = [x in top_vk for x in data['vk']]
    is_top_item = [x in top_item for x in data['itemid']]

    sample_data = data.loc[np.logical_and(is_top_vk, is_top_item)]

    return sample_data
