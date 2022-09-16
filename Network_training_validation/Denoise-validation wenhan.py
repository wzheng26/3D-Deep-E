#!/usr/bin/env python
# coding: utf-8

# In[2]:


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
import h5py

from keras.models import load_model
model = tf.keras.models.load_model('C:/Users/photoacoustic/Documents/Wenhan/uint/net12modelFDUNET_03142022_add_WN_6_9_12_free_enlarged_EMI.h5')
# model = tf.keras.models.load_model('C:/Users/photoacoustic/Documents/Huijuan/denoise/uint/net001modelFDUNET_01182022.h5');
import scipy.io as sio
# TRAIN_PATH = 'C:/Users/photoacoustic/Desktop/wenhan/U_net_input/validation_in_vivo_breast_L2_3D_focal_weighting_normalized/L2_breast_'
TRAIN_PATH = 'C:/Users/photoacoustic/Desktop/wenhan/U_net_input/validation_breast_data/032_3D_129_256_normalized/L2_breast_'

T_total = np.zeros((430, 256, 256, 1), dtype=np.uint8)

jj = 1

SUFFIX = '.mat'
# for j in range(1,250):
for j in range(1,430):
    
     
     xpath = TRAIN_PATH
     img =sio.loadmat(xpath + str(j) +SUFFIX)
     #img =sio.loadmat(xpath +SUFFIX)
     img = img['img'] 
     # img = img['img'] 
     img = resize(img, (256, 256), mode='constant', preserve_range=True)
     img = np.expand_dims(img, axis=2)
     T_total[jj] = img
     jj = jj + 1
        
preds_test = model.predict(T_total, verbose=1)

import numpy as np

from scipy import io
io.savemat('output_test0405_032_3D_normalized.mat', {'output': preds_test})
io.savemat('input_test0405_032_3D_normalized.mat', {'input': T_total})
#io.savemat('target_test.mat', {'target': Y_total})


# In[ ]:


import matplotlib.pyplot as plt  
#for i in range(1,256):
fig = plt.figure()
ax = fig.add_subplot(1, 3, 1)
plt.imshow(np.squeeze(T_total[150,:,:,]))
ax = fig.add_subplot(1, 3, 2)
plt.imshow(np.squeeze(preds_test[150,:,:,]))
#ax = fig.add_subplot(1, 3, 3)
#plt.imshow(np.squeeze(Y_total[1,:,:,]))

