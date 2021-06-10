import numpy as np
import tensorflow as tf
from tensorflow import keras

labels = np.load('processed_labels.npy')
data = np.load('processed_data.npy')