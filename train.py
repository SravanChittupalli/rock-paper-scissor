# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:45:04 2020

@author: Sarvan
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import os
import numpy as np
import cv2
import random

IMAGE_FOLDER = 'image_data'
CLASS_VAL={
        'rock':0,
        'paper':1,
        'scissor':2,
        'none':3
        }

def mapper(val):
    return CLASS_VAL[val]

dataset=[]
for directory in os.listdir(IMAGE_FOLDER):
    img_folder= os.path.join(IMAGE_FOLDER,directory)
    for image in os.listdir(img_folder):
        img = cv2.imread(os.path.join(img_folder , image))
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        dataset.append([img,directory])

def get_model():
    model=tf.keras.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), input_shape=(200,200,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512 , activation='relu'),
            tf.keras.layers.Dense(4 , activation='softmax')         
            ])
    return model


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy') > .990):
            print("MODEL IS OUTPUTTING AN ACCURACY OF 99%...HENCE STOPPING TRAINING")
            self.model.stop_training=True
callbacks = myCallback()

random.shuffle(dataset)
# * shows unzipping. While zip() combines tuples,dict,list etc,
# while zip(*   ) seperates them
datas , label = zip(*dataset)

#converts tuple to ndarray
data = np.asarray(datas)


# what is map doing over here?
#map(function , list)
#label is sent to mapper function which sends the value from the dictionary
label = list(map(mapper , label))
        
label = keras.utils.to_categorical(label)
print(label)

label = np.asarray(label)       

model = get_model()                
model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
model.fit(data , label , epochs=6 , steps_per_epoch = 10 , validation_split=.1 , callbacks=[callbacks])
model.save("rock-paper-scissor.h5")    