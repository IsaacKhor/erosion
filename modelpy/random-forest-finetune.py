import sys, os, cv2, sklearn
import sklearn.model_selection, sklearn.ensemble
import numpy as np

raw_data = np.load('processed/summary-extended.npy')
with open('pylabels.txt', 'r') as f:
    lns = f.readlines()
    sim_labels = eval(lns[0])

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

for sim_no in range(1,2000):
    # print('Loading: sim', sim_no)
    category = label_to_cat(sim_labels[sim_no-1])
    if category == 2:
        continue

    for step in range(30,200):
        img = raw_data[sim_no, step]
        img = img.reshape(50,50,1)
        avg_dark_data.append(avg_row_dark(img))
        labels.append(category)

print('Done with loading data')

# convert everything into numpy-land
avg_dark_data = np.asarray(avg_dark_data)
labels = np.asarray(labels)

# Split into testing/training sets
train_x, test_x, train_y, test_y = sklearn.model_selection.train_test_split(
    avg_dark_data, labels, test_size=0.3, random_state=42)

# Fine tuning
# ===========

n_estimators = [int(x) for x in np.linspace(50, 500, 11)]
max_depth = [int(x) for x in np.linspace(10, 200, 40)]
min_samples_split = np.linspace(0.0001, 0.01, 100)
max_features = ['sqrt', 'log2']
bootstrap = [True, False]

random_grid = {
    'n_estimators': n_estimators,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'max_features': max_features,
}

from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as sm

rf = sklearn.ensemble.RandomForestClassifier()
model_rand = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=100,
    cv=4,
    verbose=2,
    random_state=42,
    n_jobs=-1)
# model_rand.fit(avg_dark_data, labels)
# print(model_rand.best_params_)
# model_rand.best_estimator_.save('models/rf-finetune.hdf5')
# Best params
# {'n_estimators': 140, 'min_samples_split': 0.0079, 'max_features': 'sqrt', 'max_depth': 10}

model = sklearn.ensemble.RandomForestClassifier(
    n_estimators=140,
    max_depth=10,
    min_samples_split=0.0079,
    max_features='sqrt')


# Evaluation
# ==========

def eval_model(model, x, y):
    preds = model.predict(x)
    acc = sm.accuracy_score(preds, y)
    prec = sm.precision_score(preds, y)
    return acc, prec

