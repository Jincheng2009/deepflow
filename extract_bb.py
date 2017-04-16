
# coding: utf-8

# In[47]:

from keras.models import load_model
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import selectivesearch
import random

datapath = "/home/jincheng/projects/flow_cytometry/sim_data/"

annotation = pd.read_json(os.path.join(datapath, "annotation_o10_equal.js"))

f = h5py.File(os.path.join(datapath, "sim_data_o10_equal.hdf5"), 'r')

x = []

train_idx = random.sample(range(len(annotation)), int(0.001 * len(annotation)))
annotation = annotation.loc[train_idx] 
annotation = annotation.reset_index()

for name in annotation['name']:
    dset = np.array(f[name])
    image = dset.reshape(np.append(1, dset.shape))
    x.append(image)
    
print("Complete loading")
x = np.concatenate(x, axis=0) 


# In[48]:

# draw rectangles on the original image
import matplotlib.patches as mpatches

def plot_density(img, bb=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    if bb is not None:
        for x, y, w, h in bb:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    plt.show()

idx = annotation.shape[0] - 1
bb = annotation.loc[idx]['bb']
plot_density(dset, bb)


# In[49]:

print(bb[0])
x, y, w, h = map(lambda x : int(round(x)), bb[0])
x, y = max(x, 0), max(y, 0)
w, h = min(128 - x, w), min(128 - y, h)
print(str(x))
pop_img = dset[y:y+h, x:x+w, :]
print(pop_img.shape)
plt.imshow(pop_img)
plt.show()


# In[50]:

import tensorflow as tf
sess = tf.Session()
pop_img1 = tf.image.resize_images(pop_img, (64,64)).eval(session = sess)
print(pop_img1.shape)
print(pop_img1[:5,:5,0])
plt.imshow(pop_img1)
plt.show()
#tf.image.resize_images(images, size, method=ResizeMethod.BILINEAR, align_corners=False)


# In[57]:

import selectivesearch

img_lbl, regions = selectivesearch.selective_search(dset, scale=500, sigma=0.9, min_size=10)
print(regions[:10])

candidates = set()
for r in regions:
    # excluding same rectangle (with different segments)
    if r['rect'] in candidates:
        continue
    # excluding regions smaller than 2000 pixels
    if r['size'] < 100:
        continue
    # distorted rects
    x, y, w, h = r['rect']
    if w / h > 5 or h / w > 5:
        continue
    candidates.add(r['rect'])

plot_density(dset, candidates)


# In[58]:

# bg_bb = np.array(annotation['bb'])
bg_bb = np.concatenate(annotation['bb'], axis=0)
print(bb)

plot_density(dset, bg_bb)


# In[59]:

print(bg_bb.shape)


# In[71]:

def area(a):
    return(a[2] * a[3])
    
def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return(0,0,0,0)
    return (x, y, w, h)

def iou(bb1, bb2):
    intersectiona = area(intersection(bb1, bb2))
    uniona = area(bb1) + area(bb2) - intersectiona
    return intersectiona / uniona

print(bb[0])
overlap = map(lambda x : iou(x, bb[0]), bg_bb)
print(overlap)

bg_idx = np.where(np.array(overlap) < 0.3)
bg_idx1 = np.where(np.array(overlap) > 0.001)
bg_idx = np.intersect1d(gb_idx, gb_idx1)
print(bg_idx2)


# In[72]:

plot_density(dset, bg_bb[bg_idx2])


# In[ ]:



