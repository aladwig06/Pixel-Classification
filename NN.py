# -*- coding: utf-8 -*-
"""
Created on Mon Oct 03 12:04:33 2016

@author: aladwig
"""
import os
#os.environ['THEANO_FLAGS'] = 'device=cpu, optimizer=None'
from keras.layers import Activation, Dense, Dropout
from keras.models import Sequential
from sklearntesting import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split

def model():
    
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=25))
    model.add(Activation('relu'))
    model.add(Dense(output_dim=2))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def load_data():
    x, y = load_train_data()
    y = np.reshape(y,[-1,1])
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    trX, teX, trY, teY = train_test_split(x,y, test_size=0.2)
    return trX, teX, trY, teY

    

model = model()
trX, teX, trY, teY = load_data()
model.fit(trX, trY, nb_epoch=1)
#metrics = model.evaluate(teX, teY)
test_img = load_test_imgs()
results = model.predict_classes(test_img)
out = []
for i in range(len(results)):
    if i > 0 :
        out.append(255)
    else:
        out.append(0)
out = np.array(out)
out = np.reshape(out, [64,80])
cv2.imshow('',out)
cv2.waitKey()
       