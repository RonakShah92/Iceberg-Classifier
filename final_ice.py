# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
py.init_notebook_mode(connected=True) # this is for 3-d image. This requires IPython
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

#Import Keras.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping


# importing data
#df = pd.read_json('train.json')
df = pd.read_json('Project/Iceberg/works/data/processed/train.json')

#------------------------------------------------------------------------------
# FUNCTIONS
#-----------------------------------------------------------------------------
# function - mapping
# convert the value in original interval to value in target interval
def mapping(n, input_min, input_max, target_min, target_max):
    return (n - input_min) / (input_max - input_min) * (target_max - target_min)

#--------------------------------------------------------------
# Function 1 - Peaks
def findMax(arr):
    arr = np.array(arr)
    n = int(np.sqrt(len(arr)))
    if len(arr.shape) == 2:
        n = len(arr)
        arr = np.array(arr, (1,(len(arr),len(arr[0]))))
    MAX = np.max(arr)
    idxMax = pd.DataFrame(arr).idxmax().iloc[0]
    idxMax = (int(idxMax / n), idxMax % n)
    return MAX, idxMax

def upperThree(i,j, length = 75):
    if i == 0:
        return[]
    elif j == 0:
        return [(i-1, j),(i-1, j+1)]
    elif j == length - 1:
        return [(i-1, j - 1),(i-1,j)]
    else:
        return [(i-1, j-1),(i-1, j),(i-1, j+1)]
    
def lowerThree(i,j, length = 75):
    if i == length - 1:
        return []
    elif j == 0:
        return [(i + 1, j),(i+1, j+1)]
    elif j == length - 1:
        return [(i+1, j-1),(i+1,j)]
    else:
        return [(i+1, j-1),(i+1, j),(i+1, j+1)]
    
def left(i,j):
    if j == 0:
        return []
    else:
        return [(i, j - 1)]
def right(i,j, length):
    if j == length - 1:
        return []
    else:
        return [(i, j + 1)]

# this returns the list of 8 index points near the target index (i,j)
def findNeighbors(i,j, length = 75):
    return upperThree(i,j, length)+ left(i,j) + right(i,j, length) + lowerThree(i,j, length)

# this function checks if the target point (i,j) is the peak (max among 3x3 matrix)
def isPeak(i,j,arr):
    length = len(arr)
    neighbors = findNeighbors(i,j, length)
    neighborVals = [arr[i][j] for i,j in neighbors]
    if arr[i][j] >= max(neighborVals):
        return 1
    else:
        return 0

def getPeaks(arr, num_largest = 50):
    length = int(np.sqrt(len(arr)))
    list_index = []
    list_val = []
    tmp = pd.DataFrame(arr)
    nlargest = tmp[0].nlargest(num_largest)
    arr = np.reshape(arr, (length,length))
    for j in nlargest.index:
        x,y = (int(j/length), j%length)
        if isPeak(x,y,arr):
            list_index.append((x,y))
            list_val.append(arr[x][y])
    del tmp
    return list_index, list_val

def getDistance(a,b):
    return ((a[0] - b[0])**2 + (a[1] - b[1]) ** 2) ** (1/2)

def numOfPeaks(arr, thres):
    arr = np.array(arr)
    if len(arr.shape) == 2:
        arr = np.reshape(arr, ((len(arr) * len(arr))),1)
    peaks = getPeaks(arr, 100)
    a = peaks[0][0] # global peak
    count = 0
    list_index = []
    for i in peaks[0]:
        if getDistance(a, i) < thres:
            count += 1
            list_index.append(i)
    return count, list_index
#-----------------------------------------------------------

#-----------------------------------------------------------
# Function 2 - convolution
def conv(arr, kernal):
    arr = np.array(arr)
    if len(arr.shape) == 1:
        arr = np.reshape(arr, (int(len(arr) / np.sqrt(len(arr))), int(len(arr)/np.sqrt(len(arr)))))
    tmp = np.zeros((len(arr) + len(kernal)-1, len(arr[0]) + len(kernal[0])-1))
    new_arr = arr.copy()
    tmp[1:1+len(arr), 1:1+len(arr)] = arr
    for i in range(len(tmp) - len(kernal)+1):
        for j in range(len(tmp[0])-len(kernal)+1):
            aval = np.sum((tmp[i:i + len(kernal), j:j + len(kernal)]) * (kernal))
            new_arr[i,j] = aval
    return new_arr

# convolution given 1. flattend array, 2. list of kernals, 3. number of convolutions
def convolution(arr, list_kernal, n = 1):
    arr = np.array(arr)
    if len(np.array(arr).shape) == 1:
        arr = np.reshape(arr, (int(np.sqrt(len(arr))),int(np.sqrt(len(arr)))))
    if n == 0:
        return arr    
    new_arr = np.zeros((len(arr),len(arr[0])))
    for i in range(n):
        MIN = np.min(arr)
        MAX = np.max(arr)
        arr = [mapping(i, MIN, MAX, 0, (MAX - MIN)) for i in arr]
        for j in range(len(list_kernal)):
            new_arr = conv(arr, list_kernal[j]) / np.sum(list_kernal[j])
        arr = new_arr
    arr = arr[n:len(arr) - n, n: len(arr) - n]
    return arr

# normalizing method - replace i,j with mean of 3x3 around i,j
def normalization(arr):
    arr = np.array(arr)
    n = len(arr)
    if len(arr.shape) == 1:
        n = int(np.sqrt(len(arr)))
        arr = np.reshape(arr, (n,n))
    new_arr = np.zeros((n-2,n-2))
    for i in range(1, n-1):
        for j in range(1, n-1):
            neighbors = findNeighbors(i,j, n)
            neighborVals = [arr[i][j] for i,j in neighbors]
            new_arr[i-1][j-1] = np.mean(neighborVals)
    return new_arr
            

#------------------------------------------------------

# Data exploration
#---------------------------------------
# 1. Expanding features
# Adding simple numerical information
# Sum(band_1, band_2)
df['band_1+band_2'] = [np.array(i) + np.array(j) for i,j in zip(df['band_1'],df['band_2'])]

# Adding mean
df['band_1 (mean)'] = [np.mean(i) for i in df['band_1']]
df['band_2 (mean)'] = [np.mean(i) for i in df['band_2']]
df['band_1&2 (mean)'] = [np.mean(i) for i in df['band_1+band_2']]

# Adding min
df['band_1 (min)'] = [np.min(i) for i in df['band_1']]
df['band_2 (min)'] = [np.min(i) for i in df['band_2']]
df['band_1&2 (min)'] = [np.min(i) for i in df['band_1+band_2']]

# Adding std
df['band_1 (std)'] = [np.std(i) for i in df['band_1']]
df['band_2 (std)'] = [np.std(i) for i in df['band_2']]
df['band_1&2 (std)'] = [np.std(i) for i in df['band_1+band_2']]

# Adding max
df['band_1 (max)'] = [np.max(i) for i in df['band_1']]
df['band_2 (max)'] = [np.max(i) for i in df['band_2']]
df['band_1&2 (max)'] = [np.max(np.array(i) + np.array(j)) for i, j in zip((df['band_2']), (df['band_1']))]

# Adding numOfPeaks
df['band_1&2 (numOfPeaks)'] = [numOfPeaks(i, 100)[0] for i in df['band_1+band_2']]

# check correlations 
df.corr()

# find difference between iceberg and boat with their statistics.
df[df['is_iceberg'] == 1][['band_1&2 (max)', 'band_1&2 (min)', 'band_1&2 (mean)']].describe()
df[df['is_iceberg'] == 0][['band_1&2 (max)', 'band_1&2 (min)', 'band_1&2 (mean)']].describe()
# plotting..
df[df['is_iceberg'] == 1]['band_1&2 (max)'].hist()
#plt.savefig('ice_hist(max)')
plt.show()
df[df['is_iceberg'] == 0]['band_1&2 (max)'].hist()
#plt.savefig('boat_hist(max)')
plt.show()

#-------------------------------------------------
# 2. Data Transformation using convolution
# using this kernal...
kernal1 = np.reshape(np.array([3,3,3,
                              3,1,3,
                              3,3,3]),
                    (3,3))
    
list_kernal = [kernal1]
n = 1
df['band(convolutioned)'] = [np.reshape(convolution(i, list_kernal,n), (1,(75-2*n)*(75-2*n)))[0] for i in df['band_1+band_2']]

# adding simple numerical information based on convolutioned image
# Adding mean
df['band(convolutioned - mean)'] = [np.mean(i) for i in df['band(convolutioned)']]

# Adding min
df['band(convolutioned - min)'] = [np.min(i) for i in df['band(convolutioned)']]

# Adding std
df['band(convolutioned - std)'] = [np.std(i) for i in df['band(convolutioned)']]

# Adding max
df['band(convolutioned - max)'] = [np.max(i) for i in df['band(convolutioned)']]

# Adding numOfPeaks
df['band(convolutioned - numOfPeaks)'] = [numOfPeaks(i, 100)[0] for i in df['band(convolutioned)']]

#------------------------------------------

#---------------------------------------------------
# Error function
def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota

def accuracy(actual, predicted):
    count = 0
    for i in range(len(predicted)):
        idx = -1;
        for j in range(3):
            if predicted[i][j] == np.max(predicted[i]):
                idx = j
                break
        if idx==actual[i]:
            count += 1
    return count / len(actual)
#----------------------------------------------------------
# Testing
#======================================
#  1-a. Logistic Regression with convolutioned data
list_cols = ['band(convolutioned - mean)', 'band(convolutioned - min)', 'band(convolutioned - std)', 'band(convolutioned - max)', 'band(convolutioned - numOfPeaks)']
list_labels = ['is_iceberg']
tmp_df = df[list_cols]
y = df['is_iceberg'].values
xtrain, xvalid, ytrain, yvalid = train_test_split(tmp_df.values, y, stratify=y, random_state=42, test_size=0.1)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain, ytrain)
predictions = clf.predict_proba(xvalid)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print("accuracy: ", accuracy(yvalid, predictions))

#=============================================
# 1-b. Logistic Regression with original data
list_cols = ['band_1&2 (mean)', 'band_1&2 (min)', 'band_1&2 (std)', 'band_1&2 (max)', 'band_1&2 (numOfPeaks)']
list_labels = ['is_iceberg']
tmp_df = df[list_cols]
y = df['is_iceberg'].values
xtrain, xvalid, ytrain, yvalid = train_test_split(tmp_df.values, y, stratify=y, random_state=42, test_size=0.1)

clf = LogisticRegression(C=1.0)
clf.fit(xtrain, ytrain)
predictions = clf.predict_proba(xvalid)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print("accuracy: ", accuracy(yvalid, predictions))

#=========================================
# 2. SVM with convolutioned data
list_cols = ['band(convolutioned - mean)', 'band(convolutioned - min)', 'band(convolutioned - std)', 'band(convolutioned - max)', 'band(convolutioned - numOfPeaks)']
list_labels = ['is_iceberg']
tmp_df = df[list_cols]
y = df['is_iceberg'].values
xtrain, xvalid, ytrain, yvalid = train_test_split(tmp_df.values, y, stratify=y, random_state=42, test_size=0.1)

svd = decomposition.TruncatedSVD()
svd.fit(xtrain)
xtrain_svd = svd.transform(xtrain)
xvalid_svd = svd.transform(xtrain)
# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain)
xtrain_svd_scl = scl.transform(xtrain)
xvalid_svd_scl = scl.transform(xvalid)

clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print("accuracy: ", accuracy(yvalid, predictions))

#=====================================
# SVM - original data
list_cols = ['band_1&2 (mean)', 'band_1&2 (min)', 'band_1&2 (std)', 'band_1&2 (max)', 'band_1&2 (numOfPeaks)']
list_labels = ['is_iceberg']
tmp_df = df[list_cols]
y = df['is_iceberg'].values
xtrain, xvalid, ytrain, yvalid = train_test_split(tmp_df.values, y, stratify=y, random_state=42, test_size=0.1)

svd = decomposition.TruncatedSVD()
svd.fit(xtrain)
xtrain_svd = svd.transform(xtrain)
xvalid_svd = svd.transform(xtrain)
# Scale the data obtained from SVD. Renaming variable to reuse without scaling.
scl = preprocessing.StandardScaler()
scl.fit(xtrain)
xtrain_svd_scl = scl.transform(xtrain)
xvalid_svd_scl = scl.transform(xvalid)

clf = SVC(C=1.0, probability=True) # since we need probabilities
clf.fit(xtrain_svd_scl, ytrain)
predictions = clf.predict_proba(xvalid_svd_scl)

print ("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))
print("accuracy: ", accuracy(yvalid, predictions))


# --------------------------------------------------------------------
# Convolutional Network (CNN)

#define our model
def getModel():
    #Building the model
    gmodel=Sequential()
    #Conv Layer 1
    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 2
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 3
    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Conv Layer 4
    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    gmodel.add(Dropout(0.2))

    #Flatten the data for upcoming dense layers
    gmodel.add(Flatten())

    #Dense Layers
    gmodel.add(Dense(512))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Dense Layer 2
    gmodel.add(Dense(256))
    gmodel.add(Activation('relu'))
    gmodel.add(Dropout(0.2))

    #Sigmoid Layer
    gmodel.add(Dense(1))
    gmodel.add(Activation('sigmoid'))

    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    gmodel.compile(loss='binary_crossentropy',
                  optimizer=mypotim,
                  metrics=['accuracy'])
    gmodel.summary()
    return gmodel


def get_callbacks(filepath, patience=2):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = ModelCheckpoint(filepath, save_best_only=True)
    return [es, msave]
file_path = ".model_weights.hdf5"
callbacks = get_callbacks(filepath=file_path, patience=5)

#train = pd.read_json("train.json")
train = pd.read_json('Project/Iceberg/works/data/processed/train.json')

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis],((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]], axis=-1)
target_train=train['is_iceberg']
X_train_cv, X_valid, y_train_cv, y_valid = train_test_split(X_train, target_train, random_state=1, train_size=0.75)

#Without denoising, core features.
import os
gmodel=getModel()
gmodel.fit(X_train_cv, y_train_cv,
          batch_size=24,
          epochs=50,
          verbose=1,
          validation_data=(X_valid, y_valid),
          callbacks=callbacks)

gmodel.load_weights(filepath=file_path)
score = gmodel.evaluate(X_valid, y_valid, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
