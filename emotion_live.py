import cv2
import numpy as np
# Import the functions to calculate feature descriptors
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
from skimage.filters import gaussian
# To read file names
import argparse as ap
import glob
import os
#To Display
import matplotlib.pyplot as plt
from skimage import data, color, exposure
model_path='model_path/svm.model'



faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
video_capture = cv2.VideoCapture(0)
clf = joblib.load(model_path)

while True:
    #Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)    )


    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        crop=frame[y:y+h,x:x+w]    
    resize=cv2.resize(crop,(48,48))
    # Display the resulting frame
    im=cv2.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    im=gaussian(im)
    fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
    pred=clf.predict(fd)
    if pred==0:
        cv2.putText(frame,"Angry" ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    elif pred==1:
        cv2.putText(frame,"DISGUST" ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    elif pred==2:
        cv2.putText(frame,"FEAR", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    elif pred==3:
        cv2.putText(frame,"Happy" ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    elif pred==4:
        cv2.putText(frame,"SAD" ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    elif pred==5:
        cv2.putText(frame,"SURPRISE" ,(x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)
    elif pred==6:
        cv2.putText(frame,"NEUTRAL", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 3, 255)

    #Display Image
    cv2.imshow('Normal',frame)
    #If key pressed == q Exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
