# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 11:50:44 2019

@author: saket
"""

import pandas as pd
a=pd.read_csv('dataset_v21.csv')
b=pd.read_csv('dataset_v22.csv')
c=pd.read_csv('dataset_v23.csv')
d=pd.read_csv('dataset_v24.csv')
e=pd.read_csv('dataset_v25.csv')
f=pd.read_csv('dataset_v26.csv')
g=pd.read_csv('dataset_v27.csv')
h=pd.read_csv('dataset_v28.csv')
l=pd.read_csv('dataset_v29.csv')
m=pd.read_csv('dataset_v30.csv')

x1=a.iloc[:,1].values
x2=b.iloc[:,1].values
x3=c.iloc[:,1].values
x4=d.iloc[:,1].values
x5=e.iloc[:,1].values
x6=f.iloc[:,1].values
x7=g.iloc[:,1].values
x8=h.iloc[:,1].values
x9=l.iloc[:,1].values
x10=m.iloc[:,1].values


y1=a.iloc[:,2].values
y2=b.iloc[:,2].values
y3=c.iloc[:,2].values
y4=d.iloc[:,2].values
y5=e.iloc[:,2].values
y6=f.iloc[:,2].values
y7=g.iloc[:,2].values
y8=h.iloc[:,2].values
y9=l.iloc[:,2].values
y10=m.iloc[:,2].values

z1=a.iloc[:,-1].values
z2=b.iloc[:,-1].values
z3=c.iloc[:,-1].values
z4=d.iloc[:,-1].values
z5=e.iloc[:,-1].values

import cv2
import numpy as np


#X = np.zeros((len(x1)+len(x2)+len(x3)+len(x4)-4,240,320))
X = np.zeros((len(x1)+len(x2)+len(x3)+len(x4)+len(x5)+len(x6)+len(x7)+len(x8)+len(x9)+len(x10),310,480),dtype='uint8')
Y = np.zeros((len(x1)+len(x2)+len(x3)+len(x4)+len(x5)+len(x6)+len(x7)+len(x8)+len(x9)+len(x10),1))
Z = np.zeros((len(x1)+len(x2)+len(x3)+len(x4)+len(x5)-5,3))

i=0
for i in range(len(x1)):
    tmp=cv2.imread(x1[i],0)
    X[i] = tmp[50:,:]
    Y[i]=y1[i]
j=i+1
for i in range(len(x2)):
    tmp=cv2.imread(x2[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y2[i]

j=j+i+1

for i in range(len(x3)):
    tmp=cv2.imread(x3[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y3[i]
 
j=j+i+1

for i in range(len(x4)):
    tmp=cv2.imread(x4[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y4[i]
    
j=j+i+1

for i in range(len(x5)):
    tmp=cv2.imread(x5[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y5[i]
    
j=j+i+1

for i in range(len(x6)):
    tmp=cv2.imread(x6[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y6[i]
    
j=j+i+1

for i in range(len(x7)):
    tmp=cv2.imread(x7[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y7[i]
    
j=j+i+1

for i in range(len(x8)):
    tmp=cv2.imread(x8[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y8[i]
    
j=j+i+1

for i in range(len(x9)):
    tmp=cv2.imread(x9[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y9[i]
    
j=j+i+1

for i in range(len(x10)):
    tmp=cv2.imread(x10[i],0)
    X[i+j] = tmp[50:,:]
    Y[i+j]=y10[i]
    

'''
j=j+i+1

for i in range(len(x4)-1):
    tmp=cv2.imread(x4[i],0)
    tmp = tmp[80:,:]
    X[i+j]=cv2.Canny(tmp,100,200)
    Y[i+j]=y4[i]
    
j=j+i+1

for i in range(len(x5)-1):
    tmp=cv2.imread(x5[i],0)
    tmp = tmp[80:,:]
    X[i+j]=cv2.Canny(tmp,100,200)
    Y[i+j]=y5[i]
'''

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.1, random_state=42)

X_train = X_train / 255.0
X_test = X_test/255.0


X_train = X_train[:,:,:,np.newaxis]
X_test = X_test[:,:,:,np.newaxis]

for i in range(len(Y_train)):
    Y_train[i] = round(Y_train[i][0],2)
    
for i in range(len(Y_test)):
    Y_test[i] = round(Y_test[i][0],2)
    

'''
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
Y_train = sc.fit_transform(Y_train)
Y_test = sc.transform(Y_test)
'''
ymin = np.min(Y_train)
Y_train = Y_train - ymin
Y_test = Y_test - ymin

ymax = np.max(Y_test)

Y_train=Y_train/ymax
Y_test = Y_test/ymax


import tensorflow as tf
model1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(36,3,3,input_shape=(310,480,1),activation=tf.nn.elu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(48,3,3,activation=tf.nn.elu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64,3,3,activation=tf.nn.elu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, kernel_initializer = 'uniform', activation=tf.nn.elu),
  tf.keras.layers.Dense(50, kernel_initializer = 'uniform', activation=tf.nn.elu),
  tf.keras.layers.Dense(10, kernel_initializer = 'uniform', activation=tf.nn.elu),
  tf.keras.layers.Dense(1, kernel_initializer = 'uniform', activation=tf.nn.elu) 
])
    
model1 = tf.keras.models.Sequential([
  #tf.keras.layers.Conv2D(32,(3,3),activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.25),
  #tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
  #tf.keras.layers.Conv2D(64,(5,5),activation=tf.nn.relu),
  #tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  #tf.keras.layers.Dropout(0.25),
  #tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, kernel_initializer = 'uniform', activation=tf.nn.relu, input_dim = 51200),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(256, kernel_initializer = 'uniform', activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(512, kernel_initializer = 'uniform', activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(1, kernel_initializer = 'uniform', activation=tf.nn.relu)
])

model1.summary()    
    
model1.compile(optimizer= 'adam', loss='mse', metrics = ['accuracy'])

m = model1.fit(X_train, Y_train, epochs=40, batch_size = 16, validation_data = (X_test, Y_test), shuffle=True)

model1.save_weights('mod_3.h5')

model1.load_weights('model2.h5')
keras_file = "lane_model1.h5"
tf.keras.models.save_model(model1, keras_file)

converter = tf.lite.TFLiteConverter.from_keras_model_file("lane_model1.h5")
tflite_model = converter.convert()
open("converted_lane1.tflite", "wb").write(tflite_model)



import tensorflow as tf
model1 = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(36,3,3,input_shape=(310,480,1),activation=tf.nn.elu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(48,3,3,activation=tf.nn.elu),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(64,3,3,activation=tf.nn.elu),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(100, kernel_initializer = 'uniform', activation=tf.nn.elu),
  tf.keras.layers.Dense(50, kernel_initializer = 'uniform', activation=tf.nn.elu),
  tf.keras.layers.Dense(10, kernel_initializer = 'uniform', activation=tf.nn.elu),
  tf.keras.layers.Dense(1, kernel_initializer = 'uniform', activation=tf.nn.elu) 
])
    
model1.load_weights('mod_3.h5')

ymax = 3.21
ymin = 9.29

ind = 74
img= cv2.imread(x2[ind],0)
image = img[50:,:]
image = image[np.newaxis,:,:,np.newaxis]
image = image/255.0
ans = model1.predict(image)
ans = ans[0][0]*ymax
ans = ans+ymin
img= cv2.imread(x3[ind],0)
image = img[50:,:]
plt.imshow(image)
print(ans)

interpreter = tf.lite.Interpreter(model_path="converted_lane1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], image)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

import matplotlib.pyplot as plt
plt.plot(m.history['loss'])
plt.plot(m.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()