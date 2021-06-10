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

for sim_no in range(1,500):
    print('Loading: sim', sim_no)
    category = label_to_cat(sim_labels[sim_no-1])
    if category == 2:
        continue

    for step in range(30,200):
        img = cv2.imread(
                'processed/{}/{}.jpg'.format(sim_no,step),
                cv2.IMREAD_GRAYSCALE)
        img = img.reshape(50,50)

        avg_dark_data.append(avg_row_dark(img))
        labels.append(category)

# convert everything into numpy-land
avg_dark_data = np.asarray(avg_dark_data)
labels = np.asarray(labels)

# trainsetx, val_x, trainsety, val_y = sklearn.model_selection.train_test_split(
#     avg_dark_data, labels, test_size=0.25, random_state=42)
# train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
#     trainsetx, trainsety, test_size=0.33, random_state=43)
train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
    avg_dark_data, labels, test_size=0.3, random_state=42)

import sklearn.ensemble
model_30 = sklearn.ensemble.RandomForestClassifier(n_estimators=30)
model_50 = sklearn.ensemble.RandomForestClassifier(n_estimators=50)
model_100 = sklearn.ensemble.RandomForestClassifier(n_estimators=100)

model_30 = model_30.fit(train_x, train_y)
model_50 = model_50.fit(train_x, train_y)
model_100 = model_100.fit(train_x, train_y)

import sklearn.metrics as sm
preds_30 = model_30.predict(test_x)
preds_50 = model_50.predict(test_x)
preds_100 = model_100.predict(test_x)

print('Accuracy: 30', sm.accuracy_score(preds_30, test_y))
print('Accuracy: 50', sm.accuracy_score(preds_50, test_y))
print('Accuracy: 100', sm.accuracy_score(preds_100, test_y))

print('Precision: 30', sm.precision_score(preds_30, test_y))
print('Precision: 50', sm.precision_score(preds_50, test_y))
print('Precision: 100', sm.precision_score(preds_100, test_y))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_feature_importances(forest, path):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(50), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(50), indices)
    plt.xlim([-1, 50])
    plt.savefig(path)
    plt.close()

plot_feature_importances(model_30, 'figs/classical/rforest30_importance.png')
plot_feature_importances(model_50, 'figs/classical/rforest50_importance.png')
plot_feature_importances(model_100, 'figs/classical/rforest100_importance.png')

