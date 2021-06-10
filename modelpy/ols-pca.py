import sys
import os

import cv2
import sklearn
import sklearn.model_selection
import sklearn.decomposition as sdecomp
import sklearn.ensemble
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim_labels = []
print('[LOG] Loading labels')
with open('pylabels.txt', 'r') as f:
    lns = f.readlines()
    sim_labels = eval(lns[0])

print('[LOG] Loading data')
raw_data = np.load('processed/summary.npy')

f = None
lns = None

avg_row_weights = np.arange(1,51)
def avg_row_dark(img):
    def calc_row(row):
        return (row*avg_row_weights).sum() / row.sum()
    tf = np.max(img) - img
    return np.asarray([calc_row(r) for r in tf])

def label_to_cat(lbl):
    if lbl == [1,0]:
        return 0
    if lbl == [0,1]:
        return 1
    else:
        return 2 # neither exclusively left nor right

avg_dark_data = []
labels = []

for sim_no in range(1,1000):
    # print('Loading: sim', sim_no)
    category = label_to_cat(sim_labels[sim_no-1])
    if category == 2:
        continue

    for step in range(30,200):
        img = raw_data[sim_no, step]
        avg_dark_data.append(avg_row_dark(img))
        labels.append(category)

# convert everything into numpy-land
avg_dark_data = np.asarray(avg_dark_data)
labels = np.asarray(labels)

# PCA
pca = sdecomp.PCA(n_components=2)
pca.fit(avg_dark_data)
reduced_data = pca.transform(avg_dark_data)

# Train
train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
    reduced_data, labels, test_size=0.3, random_state=42
    )

import sklearn.linear_model
model = sklearn.linear_model.LinearRegression()
model = model.fit(train_x, train_y)

import sklearn.metrics as sm
preds = model.predict(test_x)
# print('Accuracy: ', sm.accuracy_score(preds, test_y))
# print('Precision: ', sm.precision_score(preds, test_y))

colours = ['#FF0000' if x == 0 else '#0000FF' for x in labels]
test_colours = ['#FF0000' if x == 0 else '#0000FF' for x in test_y]

# Plot n=2 PCA and output, look for possible clustering
fig = plt.figure(figsize=(9.6,4.8))
s1 = fig.add_subplot(1,2,1)
s1.scatter(reduced_data[:,0], reduced_data[:,1], c=colours, s=1)
s1.set_xlabel('pca dimension 0')
s1.set_ylabel('pca dimension 1')
s1.set_title('2D PCA distribution of values')

s2 = fig.add_subplot(1,2,2)
s2.scatter(test_x[:,0], test_x[:,1], c=test_colours, s=1)
s2.set_xlabel('pca dimension 0')
s2.set_ylabel('pca dimension 1')
s2.set_title('2D PCA distribution of test values only')

fig.savefig('figs/classical/pca2_clustering.png')
plt.close(fig)

# Test out pca with n=1 and slice by time step
# transposed_data = np.transpose(raw_data, (1,0,2,3))
# for ts in range(0,200):

