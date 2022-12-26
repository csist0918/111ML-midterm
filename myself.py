import keras
from keras import models
from keras import layers
from keras.utils import to_categorical
from keras.models import Sequential
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler 
import numpy as np 
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import json
from keras import backend as K
from tensorflow.keras.layers import BatchNormalization

from sklearn.metrics import roc_curve
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

#x_scaled =np.delete(x_scaled,[...],1)

#群組1_ Byte Histogram 刪減
x_scaled =np.delete(x_scaled,[60,129,84,118,211,140,38,79,249,73,150,88,41,42,98,132,23,184,18,58,82,107,205,78,210,92,39,61,135,96,230,76,91,30,188,147,209,136,33,156,152,176,161,208,148,214,164,146,185,212,168,
318,314,288,329,313,289,341,307,262,298,290,294,295,287,271,258,292,272,302,293,296,299,268,259,291,273,274,297,300,301,267,261,279,275,278,263,276,257,285,286,277,283,284,269,280,264,282,270,281,265,266,
542,545,595,577,536,519,576,560,551,583,590,523,565,556,544,541,585,568,524,614,
624,625,
653,651,649,647,635,634,630,629,628,627,636,638,648,650,652,659,663,666,667,669,670,672,673,674,675,676,
851,884,844,858,865,873,876,880,889,890,907,911,918,933,935,939,857,861,868,878,913,925,859,862,867,883,891,894,896,898,899,900,904,905,908,909,910,912,917,919,920,922,923,924,926,931,932,934,936,937,849,
1085,1234,1328,1495,1647,1748,1925,1996,2042,2113,2146,1111,1216,1296,1306,1359,1374,1428,1489,1502,1538,1615,1623,1655,1658,1665,1711,1746,1860,1862,1891,1909,1980,2059,2089,2148,2209,2218,2220,971,1104,1123,1146,1166,1248,1358,1458,1633,1710,1908,1931,1965,1975,2014,2044,2065,
2112,2185,1038,1077,1167,1168,1171,1219,1331,1415,1452,1582,1626,1653,1714,2124,1041,1054,1067,1072,1097,1103,1122,1159,1178,1193,1334,1783,1800,1972,1000,1004,1005,1007,1100,1139,1177,944,969,981,1019,1021,1035,1037,1081,1095,1105,1115,1116,1118,1137,1191,1288,949,987,990,993,999,1015,1050,1076,1084,1088,1098,1102,1124,1134,1140,1182,1192,1194,1195,948,953,955,957,958,960,961,965,967,
974,978,980,983,991,992,997,1001,1002,1013,1025,1027,1031,1032,1034,1045,1047,1048,1056,1062,1064,1070,1075,1080,1087,1093,1094,1099,1107,1117,1120,1127,1142,1147,1149,1151,1152,1153,1157,1163,1165,1170,1173,1180,1186,943,945,952,956,959,962,966,976,977,979,982,984,985,989,998,1010,1012,1016,1017,1018,1020,1022,1026,1028,1030,1033,1040,1042,1044,1052,1053,1057,1058,1063,1069,1078,1079,
1086,1089,1091,1092,1096,1101,1106,1108,1109,1114,1119,1121,1125,1126,1128,1129,1130,1133,1135,1136,1148,1150,1156,1164,1169,1172,1175,1176,1179,1181,1184,1187,1188,1196,1009,1145,
2302,2305,2309,2324,2330,2341,2233,2249,2251,2256,2265,2290,2297,2314,2316,2317,2327,2333,2273,2287,2311,2335,2258,2275,2277,2332,
2373,2369,2365,2366,2368,2367],1)
x_test_scaled =np.delete(x_test_scaled,[60,129,84,118,211,140,38,79,249,73,150,88,41,42,98,132,23,184,18,58,82,107,205,78,210,92,39,61,135,96,230,76,91,30,188,147,209,136,33,156,152,176,161,208,148,214,164,146,185,212,168,
318,314,288,329,313,289,341,307,262,298,290,294,295,287,271,258,292,272,302,293,296,299,268,259,291,273,274,297,300,301,267,261,279,275,278,263,276,257,285,286,277,283,284,269,280,264,282,270,281,265,266,
542,545,595,577,536,519,576,560,551,583,590,523,565,556,544,541,585,568,524,614,
624,625,
653,651,649,647,635,634,630,629,628,627,636,638,648,650,652,659,663,666,667,669,670,672,673,674,675,676,
851,884,844,858,865,873,876,880,889,890,907,911,918,933,935,939,857,861,868,878,913,925,859,862,867,883,891,894,896,898,899,900,904,905,908,909,910,912,917,919,920,922,923,924,926,931,932,934,936,937,849,
1085,1234,1328,1495,1647,1748,1925,1996,2042,2113,2146,1111,1216,1296,1306,1359,1374,1428,1489,1502,1538,1615,1623,1655,1658,1665,1711,1746,1860,1862,1891,1909,1980,2059,2089,2148,2209,2218,2220,971,1104,1123,1146,1166,1248,1358,1458,1633,1710,1908,1931,1965,1975,2014,2044,2065,
2112,2185,1038,1077,1167,1168,1171,1219,1331,1415,1452,1582,1626,1653,1714,2124,1041,1054,1067,1072,1097,1103,1122,1159,1178,1193,1334,1783,1800,1972,1000,1004,1005,1007,1100,1139,1177,944,969,981,1019,1021,1035,1037,1081,1095,1105,1115,1116,1118,1137,1191,1288,949,987,990,993,999,1015,1050,1076,1084,1088,1098,1102,1124,1134,1140,1182,1192,1194,1195,948,953,955,957,958,960,961,965,967,
974,978,980,983,991,992,997,1001,1002,1013,1025,1027,1031,1032,1034,1045,1047,1048,1056,1062,1064,1070,1075,1080,1087,1093,1094,1099,1107,1117,1120,1127,1142,1147,1149,1151,1152,1153,1157,1163,1165,1170,1173,1180,1186,943,945,952,956,959,962,966,976,977,979,982,984,985,989,998,1010,1012,1016,1017,1018,1020,1022,1026,1028,1030,1033,1040,1042,1044,1052,1053,1057,1058,1063,1069,1078,1079,
1086,1089,1091,1092,1096,1101,1106,1108,1109,1114,1119,1121,1125,1126,1128,1129,1130,1133,1135,1136,1148,1150,1156,1164,1169,1172,1175,1176,1179,1181,1184,1187,1188,1196,1009,1145,
2302,2305,2309,2324,2330,2341,2233,2249,2251,2256,2265,2290,2297,2314,2316,2317,2327,2333,2273,2287,2311,2335,2258,2275,2277,2332,
2373,2369,2365,2366,2368,2367],1)

#群組2_ByteEntropy Histogram 刪減
#x_scaled =np.delete(x_scaled,[318,314,288,329,313,289,341,307,262,298,290,294,295,287,271,258,292,272,302,293,296,299,268,259,291,273,274,297,300,301,267,261,279,275,278,263,276,257,285,286,277,283,284,269,280,264,282,270,281,265,266],1)
#x_test_scaled =np.delete(x_test_scaled,[318,314,288,329,313,289,341,307,262,298,290,294,295,287,271,258,292,272,302,293,296,299,268,259,291,273,274,297,300,301,267,261,279,275,278,263,276,257,285,286,277,283,284,269,280,264,282,270,281,265,266],1)

#群組3_String 刪減
#x_scaled =np.delete(x_scaled,[542,545,595,577,536,519,576,560,551,583,590,523,565,556,544,541,585,568,524,614],1)
#x_test_scaled =np.delete(x_test_scaled,[542,545,595,577,536,519,576,560,551,583,590,523,565,556,544,541,585,568,524,614],1)

#群組4_General 刪減
#x_scaled =np.delete(x_scaled,[624,625],1)
#x_test_scaled =np.delete(x_test_scaled,[624,625],1)

#群組5_Header 刪減
#x_scaled =np.delete(x_scaled,[653,651,649,647,635,634,630,629,628,627,636,638,648,650,652,659,663,666,667,669,670,672,673,674,675,676],1)
#x_test_scaled =np.delete(x_test_scaled,[653,651,649,647,635,634,630,629,628,627,636,638,648,650,652,659,663,666,667,669,670,672,673,674,675,676],1)

#群組6_Section 刪減
#x_scaled =np.delete(x_scaled,[851,884,844,858,865,873,876,880,889,890,907,911,918,933,935,939,857,861,868,878,913,925,859,862,867,883,891,894,896,898,899,900,904,905,908,909,910,912,917,919,920,922,923,924,926,931,932,934,936,937,849],1)
#x_test_scaled =np.delete(x_test_scaled,[851,884,844,858,865,873,876,880,889,890,907,911,918,933,935,939,857,861,868,878,913,925,859,862,867,883,891,894,896,898,899,900,904,905,908,909,910,912,917,919,920,922,923,924,926,931,932,934,936,937,849],1)

#群組7_Import 刪減

#群組8_Export 刪減
#x_scaled =np.delete(x_scaled,[2302,2305,2309,2324,2330,2341,2233,2249,2251,2256,2265,2290,2297,2314,2316,2317,2327,2333,2273,2287,2311,2335,2258,2275,2277,2332],1)
#x_test_scaled =np.delete(x_test_scaled,[2302,2305,2309,2324,2330,2341,2233,2249,2251,2256,2265,2290,2297,2314,2316,2317,2327,2333,2273,2287,2311,2335,2258,2275,2277,2332],1)

#群組9_Directory 刪減
#x_scaled =np.delete(x_scaled,[2373,2369,2365,2366,2368,2367],1)
#x_test_scaled =np.delete(x_test_scaled,[2373,2369,2365,2366,2368,2367],1)

print(x_scaled.shape)

# 最後再抓取 validation

x_validation = x_scaled[480000:]
y_validation = y_train[480000:]

#scaler1 = preprocessing.StandardScaler().fit(x_validation)
#x_validation_scaled = scaler1.transform(x_validation)


network = Sequential()

network.add(layers.Dense(1500,activation='relu',input_shape=(1893,)))
#network.add(BatchNormalization())
network.add(layers.Dropout(0.02))

network.add(layers.Dense(1250,activation='relu'))
#network.add(BatchNormalization())
network.add(layers.Dropout(0.02))

network.add(layers.Dense(1000,activation='relu'))
#network.add(BatchNormalization())
network.add(layers.Dropout(0.02))

network.add(layers.Dense(750,activation='relu'))
#network.add(BatchNormalization())
network.add(layers.Dropout(0.02))

network.add(layers.Dense(500,activation='relu'))
#network.add(BatchNormalization())
network.add(layers.Dropout(0.02))

network.add(layers.Dense(250,activation='relu'))
#network.add(BatchNormalization())
network.add(layers.Dropout(0.02))

network.add(layers.Dense(1, activation='sigmoid'))
opt = keras.optimizers.Adam(learning_rate=0.001)
network.compile(optimizer=opt,
                loss='binary_crossentropy',
                metrics=['accuracy',f1_m,precision_m, recall_m]) 


network.summary()


history = network.fit(x_scaled,
                    y_train,
                    epochs=10,
                    batch_size=256,
                    validation_data=(x_validation ,y_validation))
#history_dict = history.history


#train_acc = history_dict['accuracy']
#epochs = range(1,len(train_acc)+1)
#plt.plot(epochs,train_acc,'b',label='Accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('ACC')
#plt.show()

loss, accuracy, f1_score, precision, recall = network.evaluate(x_test_scaled,y_test)
print('Loss:',loss)
print('Accuracy:',accuracy)
print('F1:',f1_score)
print('Precision:',precision)
print('Recall:',recall)

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
auc_keras = auc (fpr_keras,tpr_keras)
print('AUC:',auc_keras)
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