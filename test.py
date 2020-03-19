# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:00:45 2020

@author: Sarvan
"""
import tensorflow as tf
import keras
from keras.models import load_model
import cv2
import numpy as np
import os


REV_CLASS_MAP = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "none"
}


def mapper(val):
    return REV_CLASS_MAP[val]

start=False
model = tf.keras.models.load_model("rock-paper-scissor.h5",compile=False)
capture = cv2.VideoCapture(0)
while True:
    ret,frame = capture.read()
    cv2.rectangle(frame , (100,100) , (300,300) , (255,255,255) , 2)
    cv2.imshow("Collecting Images" , frame)
    
    if start:
        roi = frame[100:300 , 100:300]
        save_path = '1.jpg'
        cv2.imwrite(save_path , roi)
    key = cv2.waitKey(33)
    if key == ord('q'):
        break
    elif key==ord('s'):
        start = True
       

capture.release()
cv2.destroyAllWindows()
        
# prepare the image
img = cv2.imread(save_path)
img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
#img = cv2.resize(1,200,200,3)
img = np.asarray(img)
print(type(img))
img=img.reshape(1,200,200,3)

# predict the move made
pred = model.predict(np.asarray(img))
print(pred)
move_code = np.argmax(pred)
move_name = mapper(move_code)

print("Predicted: {}".format(move_name))
