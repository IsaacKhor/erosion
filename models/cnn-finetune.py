import sys
import os

import cv2
import sklearn
import sklearn.model_selection
import numpy as np

import keras
import keras.layers as klayers
import keras.models as kmodels
import keras.preprocessing.image as kpimg
import keras.callbacks.callbacks as kcallb

DROPOUT = 0.1

def build_model():
    # Based on VGGNet
    classifier = kmodels.Sequential()

    # Conv -> RELU -> Norm -> Pool -> dropout
    # At the end, we have 25x25
    classifier.add(klayers.Convolution2D(
        filters=32,
        kernel_size=(3,3),
        padding='same',
        data_format='channels_last',
        input_shape=(50,50,1),
        activation='relu'
        ))

    classifier.add(klayers.BatchNormalization(axis=-1))
    classifier.add(klayers.MaxPooling2D(pool_size=(2,2),strides=2))
    classifier.add(klayers.Dropout(DROPOUT))

    # 1 convolution layers into pool
    classifier.add(klayers.Convolution2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        ))
    classifier.add(klayers.BatchNormalization(axis=-1))

    # classifier.add(klayers.Convolution2D(
    #     filters=64,
    #     kernel_size=(3,3),
    #     activation='relu',
    #     padding='same',
    #     ))
    # classifier.add(klayers.BatchNormalization(axis=-1))

    classifier.add(klayers.MaxPooling2D(pool_size=(2,2),strides=2))
    classifier.add(klayers.Dropout(DROPOUT))

    # flatten to 1d, then into 2 dense layers
    classifier.add(klayers.Flatten())
    classifier.add(klayers.Dense(64, activation='relu'))
    classifier.add(klayers.Dense(64, activation='relu'))
    classifier.add(klayers.Dropout(DROPOUT))

    # output layer, just 2 number signifying prob of going left/right
    classifier.add(klayers.Dense(2, activation='sigmoid'))

    classifier.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return classifier

raw_data = np.load('processed/summary-extended.npy')
with open('pylabels.txt', 'r') as f:
    lns = f.readlines()
    sim_labels = eval(lns[0])

f = None
lns = None

data = []
labels = []

for sim_no in range(0,2000):
    for step in range(0,200):
        img = raw_data[sim_no, step]
        img = img.reshape(50,50,1)
        data.append(img)
        labels.append(sim_labels[sim_no])

# convert everything into numpy-land
data = np.array(data)
labels = np.array(labels)

(train_x, test_x, train_y, test_y) = sklearn.model_selection.train_test_split(
    data, labels, test_size=0.2, random_state=42)

# let GC do its thing
data = None
labels = None

# fit model
EPOCHS = 75
BATCH_SIZE = 32
TESTNUM = 4

classifier = build_model()
history = classifier.fit(train_x, train_y,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_split = 0.2,
    callbacks=[ kcallb.ModelCheckpoint(
        'checkpoints/cnn-finetune/t' + str(TESTNUM) + '-checkpoint-{epoch:02d}-{loss:.2f}-{accuracy:.4f}.hdf5')])

classifier.save(f'models/cnn-finetune-{TESTNUM}.hdf5')

eval_result = classifier.evaluate(
    test_x, test_y,
    batch_size = BATCH_SIZE)

print('Evaluation result:', eval_result)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(f'figs/tmp/cnn-ft-hist-{TESTNUM}.png')
