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

for sim_no in range(1,100):
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

# PCA
pca = sdecomp.PCA(n_components=7)
pca.fit(avg_dark_data)
reduced_data = pca.transform(avg_dark_data)

train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
    reduced_data, labels, test_size=0.3, random_state=42)

model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=500, oob_score=True, criterion='entropy')
model = model.fit(train_x, train_y)

import sklearn.metrics as sm
preds = model.predict(test_x)
print('Accuracy: ', sm.accuracy_score(preds, test_y))
print('Precision: ', sm.precision_score(preds, test_y))

def plot_feature_importances(forest, path):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(7), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(7), indices)
    plt.xlim([-1, 7])
    plt.savefig(path)
    plt.close()

plot_feature_importances(model, 'figs/classical/rf_pca_importance.png')

