# -*- coding: utf-8 -*-
"""
Created on Sun Apr 02 11:19:52 2017

@author: Jincheng
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from itertools import product
import matplotlib


def generate_data():
    ## Generate a flow 2D image
    xmin, xmax = 0 ,100
    ymin, ymax = 0, 100

    n = 128
    centers = np.random.uniform(low=10, high=90, size=4)
    centers.sort()
    ## Double negative center
    negative_pos = centers[:2]
    ## Shuffle to make sure y is not always larger than x
    np.random.shuffle(negative_pos)

    ## Double positive center
    positive_pos = centers[2:]
    np.random.shuffle(positive_pos)

    ## Get the x and y coordinates of four populations
    x_pos = [negative_pos[0], positive_pos[0]]
    y_pos = [negative_pos[1], positive_pos[1]]
    centers = list(product(x_pos, y_pos))

    ## 0 for xy-negative
    ## 1 for x-negative, y-positive
    ## 2 for x-positive, y-negative
    ## 3 for xy-positive
    fractions = np.random.uniform(size=4)
    total = np.sum(fractions)
    fractions = fractions / total
    counts = (fractions * 20000).astype(int)

    ## Number of populations
    npop = np.random.choice(range(1,5))
    pop_idx = np.random.choice(range(4), size=npop)
    exists = np.array([False] * 4)
    exists[pop_idx] = True

    ## Correlation between x and y for four population
    correlation = np.random.normal(0, scale = 0.1, size=4)
    correlation[correlation>0.99] = 0.99
    correlation[correlation<-0.99] = -0.99

    xvars, yvars = np.random.uniform(low=5, high=35, size=[2,4])

    ## Generate four population based on the pattern below
    ## **************************
    ## **************************
    ## ****   1   ||  3   *******
    ## ***********||*************
    ## ****   0   ||  2   *******
    ## **************************
    ## **************************
    df = []
    ## Bounding box matrix
    bb = pd.DataFrame(columns=["x","y","w","h"])
    for i in range(4):
        if not exists[i]:
            continue
        count = counts[i]
        mean = centers[i]
        xvar = xvars[i]
        yvar = yvars[i]
        cov = np.sqrt(xvar) * np.sqrt(yvar) * correlation[i]
        cov = [[xvar, cov], [cov, yvar]]  # diagonal covariance
        temp = np.random.multivariate_normal(mean, cov, count)
        width = 6. * np.sqrt(xvar)
        height = 6. * np.sqrt(yvar)
        ## x is vertical, y is horizontal
        bb.loc[i] = [mean[1], mean[0], height, width]
        if len(df) == 0:
            df = temp
        else:
            df = np.concatenate((df, temp), axis=0)

    img, xedges, yedges = np.histogram2d(df[:,0], df[:,1], bins = n, range=[[xmin, xmax],[ymin, ymax]])

    ## Rescale bounding box
    bb['w'] = bb['w'] / (xmax - xmin) * n
    bb['h'] = bb['h'] / (ymax - ymin) * n
    bb['x'] = (bb['x'] - xmin) / (xmax - xmin) * n - 0.5 * bb['w']
    bb['y'] = (bb['y'] - ymin) / (ymax - ymin) * n - 0.5 * bb['h']
    bb = bb.as_matrix()
    ## Calculate overlap between boxes
    overlap = np.zeros([bb.shape[0], bb.shape[0]])
    for i in range(0, bb.shape[0]):
        for j in range(i+1, bb.shape[0]):
            bb1 = bb[i,:]
            bb2 = bb[j,:]
            overlap[i,j] = area(intersection(bb1,bb2)) / np.min([area(bb1),area(bb2)])
    result = {"image": img, "bb": bb, "count":counts[pop_idx], "overlap":overlap}
    return result
    
# draw rectangles on the original image
def plot_density(img, bb=None):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    if bb is not None:
        for x, y, w, h in bb:
            rect = mpatches.Rectangle(
                (x, y), w, h, fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
    plt.show()

def area(a):
    return(a[2] * a[3])
    
def intersection(a,b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0]+a[2], b[0]+b[2]) - x
    h = min(a[1]+a[3], b[1]+b[3]) - y
    if w<0 or h<0: return(0,0,0,0)
    return (x, y, w, h)

def union(a,b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  return (x, y, w, h)
