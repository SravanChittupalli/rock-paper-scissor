# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 18:51:28 2020

@author: Sarvan
"""
description='''Script to gather data images with a particular label.

Usage: python gather_images.py <label_name> <num_samples>

The script will collect <num_samples> number of images and store them
in its own directory.

Only the portion of the image within the box displayed
will be captured and stored.

Press 'a' to start/pause the image collecting process.
Press 'q' to quit.'''

import cv2
import os
import sys

try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
    print('Lets Start')
except:
    print('insufficient arguments')
    print(description)
    exit(-1)

IMAGE_SAVE_PATH = 'image_data'
IMAGE_CLASS_PATH = os.path.join(IMAGE_SAVE_PATH , label_name)

try:
    os.mkdir(IMAGE_SAVE_PATH)
except FileExistsError:
    pass
try:
    os.mkdir(IMAGE_CLASS_PATH)
except FileExistsError:
    print("{} directory already exists".format(IMAGE_CLASS_PATH))

capture = cv2.VideoCapture(0)

start = False
count = 0

while True:
    ret , frame = capture.read() #capture frame by frame
    #ret shows if reading was successful and frame gives the pixel values
    
    if count == num_samples:
        break
    
    cv2.rectangle(frame , (100,100) , (300,300) , (255,255,255) , 2)
    #  frame  , start point ,  endpoint, color of border , thickness
    
    if start:
        roi = frame[100:300 , 100:300]
        save_path = os.path.join(IMAGE_CLASS_PATH , '{}.jpg'.format(count+1))
        cv2.imwrite(save_path , roi)
        count = count + 1
    
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame , "Collecting {}".format(count), (5,50), font ,
                0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Collecting Images" , frame)
    
    key = cv2.waitKey(33)
    if key == ord('q'):
        break
    elif key==ord('s'):
        start = True

print("SAVING COLLECTED PICTURES TO {}".format(IMAGE_CLASS_PATH))
capture.release()
cv2.destroyAllWindows()