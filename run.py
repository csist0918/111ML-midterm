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

from sklearn.metrics import auc
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import roc_auc_score, roc_curve


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


x_validation = x_scaled[480000:]
y_validation = y_train[480000:]

  
network = Sequential()
network.add(layers.Dropout(0.2))
network.add(layers.Dense(1500,activation='relu'))
network.add(layers.Dropout(0.5))
network.add(layers.Dense(1, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.001)
network.compile(optimizer=opt,
                    loss='binary_crossentropy',
                    metrics=['accuracy',f1_m,precision_m, recall_m]) 
history = network.fit(x_scaled,
                    y_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(x_validation,y_validation))


loss, accuracy, f1_score, precision, recall = network.evaluate(x_test_scaled,y_test)
print('Loss:',loss)
print('Accuracy:',accuracy)
print('F1:',f1_score)
print('Precision:',precision)
print('Recall:',recall)


gc.collect()

# ravel() 轉換成一維陣列
y_pred_keras = network.predict(x_test_scaled).ravel()
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_keras)
auc_keras = auc (fpr_keras,tpr_keras)
print('AUC:',auc_keras)
print("ROC AUC:", roc_auc_score(y_test, y_pred_keras))




def get_fpr(y_true, y_pred):
    nbenign = (y_true == 0).sum()
    nfalse = (y_pred[y_true == 0] == 1).sum()
    return nfalse / float(nbenign)


def find_threshold(y_true, y_pred, fpr_target):
    thresh = 0.0
    fpr = get_fpr(y_true, y_pred > thresh)
    while fpr > fpr_target and thresh < 1.0:
        thresh += 0.0001
        fpr = get_fpr(y_true, y_pred > thresh)
    return thresh, fpr




threshold, fpr = find_threshold(y_test, y_pred_keras, 0.1)
fnr = (y_pred_keras[y_test == 1] < threshold).sum() / float((y_test == 1).sum())
print("Ember Model Performance at 10% FPR:")
print("Threshold: {:.3f}".format(threshold))
print("False Positive Rate: {:.3f}%".format(fpr * 100))
print("False Negative Rate: {:.3f}%".format(fnr * 100))
print("Detection Rate: {}%".format(100 - fnr * 100))
print()


threshold, fpr = find_threshold(y_test, y_pred_keras, 0.01)
fnr = (y_pred_keras[y_test == 1] < threshold).sum() / float((y_test == 1).sum())
print("Ember Model Performance at 1% FPR:")
print("Threshold: {:.3f}".format(threshold))
print("False Positive Rate: {:.3f}%".format(fpr * 100))
print("False Negative Rate: {:.3f}%".format(fnr * 100))
print("Detection Rate: {}%".format(100 - fnr * 100))
print()

threshold, fpr = find_threshold(y_test, y_pred_keras, 0.001)
fnr = (y_pred_keras[y_test == 1] < threshold).sum() / float((y_test == 1).sum())
print("Ember Model Performance at 0.1% FPR:")
print("Threshold: {:.3f}".format(threshold))
print("False Positive Rate: {:.3f}%".format(fpr * 100))
print("False Negative Rate: {:.3f}%".format(fnr * 100))
print("Detection Rate: {}%".format(100 - fnr * 100))

plt.figure(figsize=(8, 8))
fpr_plot, tpr_plot, _ = roc_curve(y_test,y_pred_keras)
plt.plot(fpr_plot, tpr_plot, lw=4, color='k')
plt.gca().set_xscale("log")
plt.yticks(np.arange(22) / 20.0)
plt.xlim([4e-5, 1.0])
plt.ylim([0.65, 1.01])
plt.gca().grid(True)
plt.vlines(fpr, 0, 1 - fnr, color="r", lw=2)
plt.hlines(1 - fnr, 0, fpr, color="r", lw=2)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
_ = plt.title("myself Model ROC Curve")
plt.show()