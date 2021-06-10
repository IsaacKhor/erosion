import sys
import os
import cv2
import sklearn
import sklearn.model_selection
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

split_data = []
labels = []

def label_to_cat(lbl):
    if lbl == [1,0]:
        return 0
    if lbl == [0,1]:
        return 1
    else:
        return 2 # neither exclusively left nor right

def highest_lr_calc(img):
    left = np.asarray([(img[i,0:25] < 20).sum() for i in range(0,50)])
    right = np.asarray([(img[i,25:50] < 20).sum() for i in range(0,50)])
    return (49 - np.argmax(left > 0), 49 - np.argmax(right > 0))

for sim_no in range(1,1000):
    category = label_to_cat(sim_labels[sim_no-1])
    if category == 2:
        continue

    for step in range(25,200):
        img = raw_data[sim_no, step]
        split_data.append(highest_lr_calc(img))
        labels.append(category)

split_data = np.asarray(split_data)
split_data_noisy = np.uniform(-0.25, 0.25, split_data.shape)
colours_split = ['#FF0000' if x == 0 else '#0000FF' for x in labels]

# t=200 only, sims 0-499
ts_last = raw_data[:500,199]
data_last = np.asarray([highest_lr_calc(x) for x in ts_last])
data_noise = data_last + np.uniform(-0.25, 0.25, data_last.shape)
colours_last = ['#FF0000' if x == [1,0] else '#0000FF' for x in sim_labels[:500]]

fig = plt.figure(figsize=(9.6,4.8))
fig.suptitle('Distribution of furthest L/R (noise added to distinguish points)')
s1 = fig.add_subplot(1,2,1)
s1.scatter(split_data[:,0], split_data[:,1], c=colours, s=1)
s1.set_title('Over entire data set t>25')
s1.set_xlabel('Furtherst stream progressed on left half of image')
s1.set_ylabel('Right half of image')
s2 = fig.add_subplot(1,2,2)
s2.scatter(data_last[:,0], data_last[:,1], c=colours_last, s=1)
s2.set_title('For t=200 only')
s2.set_xlabel('Furtherst stream progressed on left half of image')
s2.set_ylabel('Right half of image')
fig.savefig('figs/classical/highest_lr_last.png')
plt.close(fig)

