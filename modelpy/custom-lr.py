import sys, os, cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

raw_data = np.load('processed/summary-extended.npy')
with open('pylabels.txt', 'r') as f:
    lns = f.readlines()
    sim_labels = eval(lns[0])

def img_to_lr_point(img):
    sum_cols = img.sum(axis=0)
    return [sum_cols[0:25].sum(), sum_cols[25:50].sum()]

TIME = 40

left_data = []
right_data = []

for i in range(1,1999):
    img = raw_data[i,TIME]
    lrpair = img_to_lr_point(img)

    if sim_labels[i] == [1,0]:
        left_data.append(lrpair)
    elif sim_labels[i] == [0,1]:
        right_data.append(lrpair)

left_data = np.array(left_data)
right_data = np.array(right_data)

# Plot

fig,ax = plt.subplots(1,1)
ax.scatter(left_data[:,0], left_data[:,1], c='#FF0000', s=2, alpha=0.4, label='Left')
ax.scatter(right_data[:,0], right_data[:,1], c='#0000FF', s=2, alpha=0.4, label='Right')
ax.set_xlabel('Sum of pixel intensities on right half of image')
ax.set_ylabel('Sum of pixel intensities on left half of image')
ax.set_title('Comparing pixel intensities on left/right half of image, t=' + str(TIME))
ax.legend()
fig.savefig(f'figs/paper/lrsum_step{TIME}.png')

