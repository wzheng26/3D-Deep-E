#!/usr/bin/env python
# coding: utf-8

# In[1]:



import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


# In[2]:


# What version of Python do you have?

import sys

import tensorflow.keras
import pandas as pd
import sklearn as sk
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print(f"Pandas {pd.__version__}")
print(f"Scikit-Learn {sk.__version__}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")


# In[3]:


# get_ipython().system('pip install libhdf5')
# get_ipython().system('pip install h5py')
# get_ipython().system('pip install cython')


# In[1]:


#!pip install scikit-image
import tensorflow as tf
import numpy as np
import scipy.io as scio
from skimage.io import imread, imshow
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import datetime
import os
from os import makedirs
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.mixed_precision import experimental as mixed_precision

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.config.optimizer.set_jit(True)
########################################################################################################################
'''initialize constants'''
########################################################################################################################
seed = 7
np.random.seed = seed
tf.random.set_seed(seed)

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 1
########################################################################################################################
'''Load dataset'''
########################################################################################################################

TRAIN_PATH = 'C:/Users/photoacoustic/Desktop/wenhan/3D focal line training/Noise_and_Data_add_noise/Data_Gaussian_12_15_18_noise_free_enlarged_EMI_source/recon_noise_IMG_'
TRAIN_PATH2 = 'C:/Users/photoacoustic/Desktop/wenhan/U_net_input/Ground_truth/1_6000_6_9_12_free_enlarged/p0_IMG_'
# data_ids = [filename for filename in os.listdir(TRAIN_PATH) if filename.startswith("sensor_")]

# NUMBER_OF_SAMPLES = int(len(data_ids))
# print(NUMBER_OF_SAMPLES)

########################################################################################################################
'''Folder for saving the model'''
########################################################################################################################
MODEL_NAME = 'modelFDUNET_03152022_add_WN_12_15_18_free_enlarged_EMI.h5'

NUMBER_OF_SAMPLES = 6000;


X_total = np.zeros((NUMBER_OF_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_total = np.zeros((NUMBER_OF_SAMPLES, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)

NUMBER_EPOCHS = 100
PATIENCE = 10
MONITOR = 'val_loss'

FOLDER_NAME = "C:/Users/photoacoustic/Documents/Wenhan/uint/net13"
makedirs(FOLDER_NAME)
MODEL_NAME = FOLDER_NAME + MODEL_NAME
LOG_NAME = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

########################################################################################################################
'''Image augmentation'''
########################################################################################################################
print('Resizing training images and masks')
import scipy.io as sio

# X_total = np.zeros((4750, 256, 256, 1), dtype=np.uint8)
# Y_total = np.zeros((4750, 256, 256, 1), dtype=np.uint8)

SUFFIX = '.mat'
for j in range(1, NUMBER_OF_SAMPLES):
    xpath = TRAIN_PATH
    img = sio.loadmat(xpath + str(j) + SUFFIX)
    img = img['pa_img']
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = np.expand_dims(img, axis=2)
    X_total[j] = img
    ypath = TRAIN_PATH2
    true_img = sio.loadmat(ypath + str(j) + SUFFIX)
    true_img = true_img['p_r_r']
    true_img = resize(true_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    true_img = np.expand_dims(true_img, axis=2)
    Y_total[j] = true_img

# true_img = scio.loadmat(ypath)#[:, :, :IMG_CHANNELS]
# true_img = true_img['p0']
# true_img = resize(true_img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
# true_img = np.expand_dims(true_img, axis=2)
# Y_total[data] = true_img

########################################################################################################################
'''Divide in training and test data'''
########################################################################################################################
test_split = 0.1
X_train, X_test, Y_train, Y_test = train_test_split(X_total, Y_total, test_size=test_split, random_state=seed)

Y_pred = np.zeros((len(X_test), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)

X_train, Y_train = shuffle(X_train, Y_train, random_state=seed)

print('Done splitting and shuffling')

########################################################################################################################
'''Network functions'''
########################################################################################################################
def Conv2D_BatchNorm(input, filters, kernel_size, strides, activation, kernel_initializer, padding):
    out = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size, strides= strides, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(input)
    out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones', beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None)(out)
    return out


def Conv2D_Transpose_BatchNorm(input, filters, kernel_size, strides, activation, kernel_initializer, padding):
    out = tf.keras.layers.Conv2DTranspose(filters, kernel_size=kernel_size, strides= strides, activation=activation, kernel_initializer=kernel_initializer, padding=padding)(input)
    out = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True,
                                            beta_initializer='zeros', gamma_initializer='ones',
                                            moving_mean_initializer='zeros',
                                            moving_variance_initializer='ones', beta_regularizer=None,
                                            gamma_regularizer=None,
                                            beta_constraint=None, gamma_constraint=None)(out)
    return out

def DownBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    shortcut = out
    out = DownSample(out, filters, kernel_size, strides=2, padding=padding,
                     activation=activation, kernel_initializer=kernel_initializer)
    return [out, shortcut]


def BrigdeBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    out = FD_Block(input, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    out = UpSample(out, filters, kernel_size, strides=2, padding=padding,
                   activation=activation, kernel_initializer=kernel_initializer)
    return out


def UpBlock(input, filters, kernel_size, padding, activation, kernel_initializer):
    out = Conv2D_BatchNorm(input, filters= filters//2, kernel_size=1, strides=1, activation=activation,
                           kernel_initializer=kernel_initializer, padding=padding)
    out = FD_Block(out, f_in=filters // 2, f_out=filters, k=filters // 8, kernel_size=3, padding='same',
                   activation=activation, kernel_initializer='glorot_normal')
    out = UpSample(out, filters, kernel_size, strides=2, padding=padding,
                     activation=activation, kernel_initializer=kernel_initializer)
    return out


def FD_Block(input, f_in, f_out, k, kernel_size, padding, activation, kernel_initializer):
    out = input
    for i in range(f_in, f_out, k):
        shortcut = out
        out = Conv2D_BatchNorm(out, filters=f_in, kernel_size=1, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = Conv2D_BatchNorm(out, filters=k, kernel_size=kernel_size, strides=1, padding=padding,
                               activation=activation, kernel_initializer=kernel_initializer)
        out = tf.keras.layers.Dropout(0.7, seed=seed)(out)
        out = tf.keras.layers.concatenate([out, shortcut])
    return out


def DownSample(input, filters, kernel_size, strides, padding, activation, kernel_initializer):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, activation= activation, kernel_initializer= kernel_initializer, padding=padding)
    out = Conv2D_BatchNorm(out, filters, kernel_size=kernel_size, strides=strides, activation=activation,
                           kernel_initializer=kernel_initializer, padding=padding)
    return out

def UpSample(input, filters, kernel_size, strides, padding, activation, kernel_initializer):
    out = Conv2D_BatchNorm(input, filters, kernel_size=1, strides=1, padding=padding,
                           activation=activation, kernel_initializer=kernel_initializer)
    out = Conv2D_Transpose_BatchNorm(out, filters//2, kernel_size=kernel_size, strides=strides, activation=activation,
                           kernel_initializer=kernel_initializer, padding=padding)
    return out




########################################################################################################################
'''Define parameters'''
########################################################################################################################
kernel_initializer = tf.keras.initializers.glorot_normal(seed=seed)
activation = 'relu'
filters = 16
padding = 'same'
kernel_size = 3
strides = 1

inputs = tf.keras.layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = inputs

out = Conv2D_BatchNorm(s, filters, kernel_size=kernel_size, strides= strides, activation=activation, kernel_initializer=kernel_initializer, padding=padding)

[out, c1] = DownBlock(out, filters*2**1, kernel_size, padding, activation, kernel_initializer)
[out, c2] = DownBlock(out, filters*2**2, kernel_size, padding, activation, kernel_initializer)
[out, c3] = DownBlock(out, filters*2**3, kernel_size, padding, activation, kernel_initializer)
[out, c4] = DownBlock(out, filters*2**4, kernel_size, padding, activation, kernel_initializer)
[out, c5] = DownBlock(out, filters*2**5, kernel_size, padding, activation, kernel_initializer)

out = BrigdeBlock(out, filters*2**6, kernel_size, padding, activation, kernel_initializer)

out = tf.keras.layers.concatenate([out, c5])
out = UpBlock(out, filters*2**5, kernel_size, padding, activation, kernel_initializer)


out = tf.keras.layers.concatenate([out, c4])
out = UpBlock(out, filters*2**4, kernel_size, padding, activation, kernel_initializer)
out = tf.keras.layers.concatenate([out, c3])
out = UpBlock(out, filters*2**3, kernel_size, padding, activation, kernel_initializer)
out = tf.keras.layers.concatenate([out, c2])
out = UpBlock(out, filters*2**2, kernel_size, padding, activation, kernel_initializer)
out = tf.keras.layers.concatenate([out, c1])

out = Conv2D_BatchNorm(out, filters, kernel_size=1, strides=1, activation=activation, kernel_initializer=kernel_initializer, padding=padding)
out = FD_Block(out, f_in=filters, f_out=filters*2, k=filters // 4, kernel_size=3, padding=padding,
                   activation=activation, kernel_initializer=kernel_initializer)

out = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding=padding, activation='linear', kernel_initializer=kernel_initializer)(out)
out = tf.keras.layers.Add()([out, s])
out = tf.keras.layers.ReLU()(out)
outputs = out
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

########################################################################################################################
'''define adam'''
########################################################################################################################
opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
model.compile(optimizer=opt, loss='mean_squared_error', metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()])

########################################################################################################################
'''Model checkpoints'''
########################################################################################################################
log_dir='.\logs'
callbacks = [tf.keras.callbacks.ModelCheckpoint(MODEL_NAME, verbose=1, save_best_only=True),
             tf.keras.callbacks.TensorBoard(log_dir=LOG_NAME),
		tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor=MONITOR)]

########################################################################################################################
'''Compile model'''
########################################################################################################################
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=5, epochs=NUMBER_EPOCHS, callbacks=callbacks)
print('Model Trained')


# In[ ]:




