import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras import backend as K 
import numpy as np 
import pandas as pd
import gc  #記憶體回收
#測試用-------------------------------------
import pickle  #資料保存模組
import json
import matplotlib.pyplot as plt #繪圖
import tensorflow as tf
from sklearn.metrics import roc_curve
#-------------------------------------------

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


train_size = 600000
test_size = 200000
columns = 2381
path = "C:\\ember\\2018"

x_train = np.memmap(path+"\\X_train.dat", dtype=np.float32, mode="r", shape=(train_size, columns))
y_train = np.memmap(path+"\\y_train.dat", dtype=np.float32, mode="r", shape=train_size)
x_test = np.memmap(path+"\X_test.dat", dtype=np.float32, mode="r", shape=(test_size, columns))
y_test = np.memmap(path+"\y_test.dat", dtype=np.float32, mode="r", shape=test_size)

 
scaler = preprocessing.StandardScaler().fit(x_train)
scaler2 = preprocessing.StandardScaler().fit(x_test)
x_scaled = scaler.transform(x_train)
x_test_scaled = scaler2.transform(x_test)


kf = KFold(7,shuffle=True) 
fold = 0

loss_group = []
accuracy_group = []
f1_score_group = []
precision_group = []
recall_group = []

for train, test in kf.split(x_scaled):
    fold += 1
    
    print(f"Fold #{fold}")
    kx_train, kx_val = x_scaled[train], x_scaled[test]
    ky_train, ky_val = y_train[train], y_train[test]
    
    network = Sequential()
    network.add(layers.Dropout(0.2))
    network.add(layers.Dense(1500,activation='relu'))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(1, activation='sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    network.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy',f1_m,precision_m, recall_m]) 
    history = network.fit(kx_train,
                    ky_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(kx_val,ky_val))

    loss, accuracy, f1_score, precision, recall = network.evaluate(x_test_scaled,y_test)
    result ='[K]:%d [Loss]:%.3f [Accuracy]:%.3f [F1]:%.3f [Precision]:%.3f [Recall]:%.3f' %(fold,loss, accuracy, f1_score, precision, recall) 
    print(result)
    loss_group.append(loss)
    accuracy_group.append(accuracy)
    f1_score_group.append(f1_score)
    precision_group.append(precision)
    recall_group.append(recall)
    # K折交叉驗證訓練時下一折前需要釋放記憶體避免記憶體被耗盡
    K.clear_session() 
    gc.collect()
    
loss_avg = np.average(loss_group)
accuracy_avg = np.average(accuracy_group)
f1_score_avg = np.average(f1_score_group)
precision_avg = np.average(precision_group)
recall_avg = np.average(recall_group)

msg2= 'Average: [Loss]:%.3f [Accuracy]:%.3f [F1]:%.3f [Precision]:%.3f [Recall]:%.3f' %(loss_avg, accuracy_avg, f1_score_avg, precision_avg, recall_avg) 
print(msg2)