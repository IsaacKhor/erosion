import sys
import os

import cv2
import sklearn
import sklearn.model_selection
import numpy as np

print('[LOG] Loading labels')
f = open('pylabels.txt', 'r')
lns = f.readlines()
f.close()

sim_labels = eval(lns[0])
# print('Sim labels', sim_labels)

f = None
lns = None

print('[LOG] Loading all images into memory')

def naive_middle_split(img):
    vertsum = img.sum(axis=0)
    return (sum(vertsum[0:25]), sum(vertsum[25:50]))

def highest_lr_calc(img):
    left = np.asarray([(img[i] < 50).sum() for i in range(0,25)])
    right = np.asarray([(img[i] < 50).sum() for i in range(25,50)])
    return (49 - np.argmax(left > 1), 49 - np.argmax(right > 1))

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

middle_split_data = []
highest_lr_data = []
avg_dark_data = []
labels = []

for sim_no in range(1,500):
    print('Loading: sim', sim_no)
    category = label_to_cat(sim_labels[sim_no-1])
    if category == 2:
        continue

    for step in range(1,200):
        img = cv2.imread(
                'processed/{}/{}.jpg'.format(sim_no,step),
                cv2.IMREAD_GRAYSCALE)
        img = img.reshape(50,50)

        middle_split_data.append(naive_middle_split(img))
        highest_lr_data.append(highest_lr_calc(img))
        avg_dark_data.append(avg_row_dark(img))
        labels.append(category)

# convert everything into numpy-land
middle_split_data = np.asarray(middle_split_data)
highest_lr_data = np.asarray(highest_lr_data)
avg_dark_data = np.asarray(avg_dark_data)
labels = np.asarray(labels)

(train_x1, test_x1, train_y1, test_y1) = sklearn.model_selection.train_test_split(
    middle_split_data, labels, test_size=0.2, random_state=42)
(train_x2, test_x2, train_y2, test_y2) = sklearn.model_selection.train_test_split(
    highest_lr_data, labels, test_size=0.2, random_state=42)
(train_x3, test_x3, train_y3, test_y3) = sklearn.model_selection.train_test_split(
    avg_dark_data, labels, test_size=0.2, random_state=42)

import sklearn.naive_bayes
import sklearn.tree
middle_split_model = sklearn.naive_bayes.GaussianNB()
highest_lr_model = sklearn.naive_bayes.GaussianNB()
avg_dark_model = sklearn.tree.DecisionTreeClassifier()

middle_split_model = middle_split_model.fit(train_x1, train_y1)
highest_lr_model = highest_lr_model.fit(train_x2, train_y2)
avg_dark_model = avg_dark_model.fit(train_x3, train_y3)

# eval_result = classifier.evaluate(
#     test_x, test_y,
#     batch_size = BATCH_SIZE)

# print('Evaluation result:', eval_result)
