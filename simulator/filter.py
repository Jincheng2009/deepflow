import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import json
import random
import h5py
import argparse

datapath = "../../sim_data"

## Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input")
parser.add_argument("-o", "--overlap", type=float)
parser.add_argument("-n", "--number", type=int)
args = parser.parse_args()

## Max overlap less than cutoff will pass
cutoff = 0.1
ntotal = 10000

if args.input:
    datapath = args.input
if args.number:
    ntotal = int(args.number)
if args.overlap:
    cutoff = args.overlap

    
annot_df = pd.read_json(os.path.join(datapath, "annotation.js"))
annot_df['max_overlap'] = map(lambda x : max(max(x)), annot_df['overlap'])
annot_df['y'] = map(lambda x : np.sum(np.array(x)), annot_df['pattern'])

filter_df = annot_df[annot_df['max_overlap'] < 0.1]
selection_df = None

random.seed(123)
for i in range(1,5):
    temp_df = filter_df[filter_df['y'] == i]
    temp_df = temp_df.reset_index()
    selection = random.sample(range(len(temp_df)), int(ntotal/4))
    temp_df = temp_df.loc[selection, :]
    if selection_df is None:
        selection_df = temp_df
    else:
        selection_df = selection_df.append(temp_df)

selection_df = selection_df.reset_index()

## Save the annotations for filtered samples
colnames = ['name', 'count', 'bb', 'pattern', 'y', 'overlap']
selection_df[colnames].to_json(os.path.join(datapath, "annotation_o10_equal.js"))

## Open input file and save the filtered data
f = h5py.File(os.path.join(datapath, "sim_data.hdf5"), "r")
newf = h5py.File(os.path.join(datapath, "sim_data_o10_equal.hdf5"), "w")

for name in selection_df['name']:
    if name in f:
        image = f[name]
        dset = newf.create_dataset(name, data = image)

print("Complete saving")
f.close()
newf.close()

