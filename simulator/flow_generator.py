# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 16:22:25 2017

@author: Jincheng
"""

from flow_lib import plot_density
from flow_lib import generate_data
import selectivesearch
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import json

n= 10 

datapath = "../data/"
data = np.empty(shape=(1,128,128,3))
annotation = []
for i in range(n):
    temp = generate_data()
    img = temp['image']
    color_map = matplotlib.cm.get_cmap('hsv')
    normed_data = (img - np.min(img)) / (np.max(img) - np.min(img))
    mapped_data = color_map(normed_data)
    temp_img = mapped_data[:,:,:3]
    temp_img = temp_img.reshape((1,128,128,3))
    data = np.append(data, temp_img, axis=0)
    temp.pop('image')
    for key in temp:
        temp[key] = temp[key].tolist()
    annotation.append(temp)

data=data[1:]

np.save(datapath + "sample.npy", data)
with open(datapath + 'annotation.js', 'w') as fp:
    json.dump(annotation, fp)
#img_lbl, regions = selectivesearch.selective_search(data, scale=500, sigma=.9, min_size=20)
#bb = [x['rect'] for x in regions if x['size'] > 20 
#      and x['rect'][2] / x['rect'][3] < 5 
#      and x['rect'][3] / x['rect'][2] < 5 ]
#plot_density(data, bb)
