import sys
import os

import cv2
import sklearn
from sklearn import model_selection, decomposition, ensemble, preprocessing
import scipy.stats
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def get_density(x,y):
    xy = np.vstack([x,y])
    z = scipy.stats.gaussian_kde(xy)(xy)
    idx = z.argsort()
    return x[idx], y[idx], z[idx]

sim_labels = []
print('[LOG] Loading labels')
with open('pylabels.txt', 'r') as f:
    lns = f.readlines()
    sim_labels = eval(lns[0])

print('[LOG] Loading data')
raw_data = np.load('processed/summary.npy')

f = None
lns = None

def stream_progression(img):
    cols_below_thres = np.asarray([(img[i] < 25).sum() for i in range(0,50)])
    first_row_from_top = np.argmax(cols_below_thres > 1)
    # Use row #4 since that's where the top "boundary" ends
    return min(first_row_from_top, 4)

avg_row_weights = np.arange(1,51)
def avg_row_dark(img):
    def calc_row(row):
        return (row*avg_row_weights).sum() / row.sum()
    tf = np.max(img) - img
    return np.asarray([calc_row(r) for r in tf])

def get_mid_of_top15(img):
    stream_row = stream_progression(img)
    # Make sure there's 15 rows *to* check
    if stream_row > 35:
        return None
    return avg_row_dark(img[stream_row:stream_row + 15])

def label_to_cat(lbl):
    if lbl == [1,0]:
        return 0
    if lbl == [0,1]:
        return 1
    else:
        return 2 # neither exclusively left nor right

avg_dark_data = []
labels = []

for sim_no in range(1,2000):
    category = label_to_cat(sim_labels[sim_no-1])
    if category == 2:
        continue
    for step in range(30,200):
        img = raw_data[sim_no, step]
        features = get_mid_of_top15(img)
        if features is None:
            continue
        avg_dark_data.append(features)
        labels.append(category)

# convert everything into numpy-land
avg_dark_data = np.asarray(avg_dark_data)
labels = np.asarray(labels)

# PCA
pca_op = decomposition.PCA()
scaled_data = preprocessing.scale(avg_dark_data)
pca_op.fit(avg_dark_data)
reduced_data = pca_op.transform(avg_dark_data)

# PCA variance
# fig, s1 = plt.subplots(1, figsize=(6,5))
# s1.plot(pca_op.explained_variance_/pca_op.explained_variance_.sum())
# s1.set_xlabel('Component #')
# s1.set_ylabel('% of explained variance')
# s1.set_title('PCA variance analysis')
# fig.savefig('figs/top15/pca_variance.png')
# plt.close(fig)

# Plot n=2 PCA and output, look for possible clustering
# fig, (s1,s2) = plt.subplots(1, 2, sharey=True, figsize=(9.6,4.8))
# left_idx = labels == 0
# right_idx = labels == 1
# x,y,z = get_density(reduced_data[left_idx,0], reduced_data[left_idx,1])
# s1.scatter(x, y, c=z, s=1)
# s1.set_xlabel('pca dimension 0')
# s1.set_ylabel('pca dimension 1')
# s1.set_title('Stream goes left')
# x,y,z = get_density(reduced_data[right_idx,0], reduced_data[right_idx,1])
# s2.scatter(x,y,c=z,s=1)
# s2.set_xlabel('pca dimension 0')
# s2.set_ylabel('pca dimension 1')
# s2.set_title('Stream goes right')
# fig.savefig('figs/top15/pca2_clustering.png')
# plt.close(fig)

# Decision tree on both raw data and PCA
raw_train_x, raw_test_x, raw_train_y, raw_test_y = sklearn.model_selection.train_test_split(
    avg_dark_data, labels, test_size=0.3, random_state=42 )

red_train_x, red_test_x, red_train_y, red_test_y = sklearn.model_selection.train_test_split(
    reduced_data, labels, test_size=0.3, random_state=42 )

# raw_forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=8)
