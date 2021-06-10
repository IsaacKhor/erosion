import sys
import os

import cv2
import numpy as np

print('[LOG] Loading labels')
f = open('pylabels.txt', 'r')
lns = f.readlines()
f.close()

sim_labels = eval(lns[0])

f = None
lns = None

print('[LOG] Loading all images into memory')

all_data = []

for sim_no in range(1,4000):
    sim_data = []
    for step in range(1,201):
        image = cv2.imread(
            'processed/{}/{}.jpg'.format(sim_no, step),
            cv2.IMREAD_GRAYSCALE)
        image = image.reshape(50,50)
        sim_data.append(image)
    all_data.append(sim_data)

all_data = np.asarray(all_data)

np.save('processed/summary-extended.npy')

