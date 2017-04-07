# -*- coding: utf-8 -*-
"""
Created on Sat Apr 01 16:22:25 2017

@author: Jincheng
"""

from flow_lib import generate_data
import matplotlib
import numpy as np
import json
import os
import h5py
import argparse

n = 10
datapath = "../data/"

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output")
parser.add_argument("-n", "--number", type=int)
args = parser.parse_args()

if args.output:
    datapath = args.output
if args.number:
    n = int(args.number)

if not os.path.exists(datapath):
    os.mkdir(datapath)
    

annotation = []

f = h5py.File(os.path.join(datapath, "sim_data.hdf5"), "w")

progress = 0.
for i in range(n):
    if float(i+1)/n >= progress:
        print("Complete " + str(float(i+1)/n * 100) + "%")
        progress += 0.05
    temp = generate_data()
    img = temp['image']
    color_map = matplotlib.cm.get_cmap('hsv')
    normed_data = (img - np.min(img)) / (np.max(img) - np.min(img))
    mapped_data = color_map(normed_data)
    temp_img = mapped_data[:,:,:3]
    name = "img" + str(i)
    dset = f.create_dataset(name, data=temp_img)
    temp.pop('image')
    temp["name"] = name
    for key in temp:
        if type(temp[key]) is np.ndarray:
            temp[key] = temp[key].tolist()
    annotation.append(temp)

print('Saving annotation data to ' + datapath)

with open(os.path.join(datapath, 'annotation.js'), 'w') as fp:
    json.dump(annotation, fp)
f.close()
#img_lbl, regions = selectivesearch.selective_search(data, scale=500, sigma=.9, min_size=20)
#bb = [x['rect'] for x in regions if x['size'] > 20 
#      and x['rect'][2] / x['rect'][3] < 5 
#      and x['rect'][3] / x['rect'][2] < 5 ]
#plot_density(data, bb)
