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

# Conv layer acting on single image
# 2 Conv2D layers with pooling and normalisation
def onestep_conv_layer():
    mdl = kmodels.Sequential()

    mdl.add(klayers.Convolution2D(
        filters=32,
        kernel_size=(3,3),
        padding='same',
        data_format='channels_last',
        input_shape=(50,50,1),
        activation='relu'
        ))
    mdl.add(klayers.BatchNormalization(axis=-1))
    mdl.add(klayers.MaxPooling2D(pool_size=(2,2),strides=2))
    mdl.add(klayers.Dropout(0.4))

    mdl.add(klayers.Convolution2D(
        filters=64,
        kernel_size=(3,3),
        activation='relu',
        padding='same',
        ))
    mdl.add(klayers.BatchNormalization(axis=-1))
    mdl.add(klayers.MaxPooling2D(pool_size=(2,2),strides=2))
    mdl.add(klayers.Dropout(0.4))

    mdl.add(klayers.Flatten())
    return mdl

def build_model():
    convlayer = onestep_conv_layer()

    mdl = kmodels.Sequential()

    #TimeDistributed layer applies the conv layer to each step individually
    mdl.add(klayers.TimeDistributed(convlayer, input_shape=(30, 50, 50, 1)))

    # GRU > LTSM, since we don't need *too* long memory and GRU performs better
    mdl.add(klayers.GRU(64))

    mdl.add(klayers.Dense(64, activation='relu'))
    mdl.add(klayers.Dense(64, activation='relu'))
    mdl.add(klayers.Dropout(0.5))

    # 2 output layers, signifying confidence in right/left
    mdl.add(klayers.Dense(2, activation='sigmoid'))
    mdl.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy',
        metrics=['accuracy']
        )

    return mdl

print('[LOG] Loading labels')
f = open('pylabels.txt', 'r')
lns = f.readlines()
f.close()

sim_labels = eval(lns[0])
print('Sim labels', sim_labels)

f = None
lns = None

print('[LOG] Loading all images into memory')

data = []
labels = []

for sim_no in range(1,2001):
    print('Loading: sim', sim_no)

    sim_batch = list()
    for step in range(30, 60):
        image = cv2.imread(
            'processed/{}/{}.jpg'.format(sim_no,step),
             cv2.IMREAD_GRAYSCALE)
        image = image.reshape(50,50,1)
        sim_batch.append(image)

    data.append(sim_batch)
    labels.append(sim_labels[sim_no-1])

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

classifier = build_model()
#history = classifier.fit(train_x, train_y,
#    epochs = EPOCHS,
#    batch_size = BATCH_SIZE,
#    validation_split = 0.2,
#    callbacks=[ kcallb.ModelCheckpoint(
#        'checkpoints/gru-batch/checkpoint-{epoch:02d}-{loss:.2f}-{accuracy:.4f}.hdf5')])

#classifier.save('models/gru-batch.hdf5')
classifier = keras.models.load_model('models/gru-batch.hdf5')

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
plt.savefig('figs/gru-batch/acc-ot-epochs.png')
