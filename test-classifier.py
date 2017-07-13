# Import the required modules
from skimage.transform import pyramid_laplacian
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
import cv2
import glob
import argparse as ap
import os
import numpy as np
#from config import *
from skimage import data, color, exposure

model_path='model_path/svm.model'
min_wdw_sz=[16, 16]
step_size=[4, 4]
orientations= 9
pixels_per_cell= [32, 32]
cells_per_block=[3, 3]

def load_path(feat_path):
    fd = joblib.load(feat_path)
    pred=clf.predict(fd)
    return pred

if __name__ == "__main__":
   
    # Load the classifier
    clf = joblib.load(model_path)

    # Load the Angry features
    angry=0
    angry_total=0
    for feat_path in glob.glob(os.path.join("feat_0","*.feat")):
        pred=load_path(feat_path)
        angry_total=angry_total+1
        if pred==0:
            angry=angry+1
    

    dis=0
    dis_t=0
    # Load the disgust features
    for feat_path in glob.glob(os.path.join("feat_1","*.feat")):
        pred=load_path(feat_path)
        dis_t=dis_t+1
        if pred==1:
            dis=dis+1
    

    # Load the fear features
    fear=0
    fear_t=0
    for feat_path in glob.glob(os.path.join("feat_2","*.feat")):
        pred=load_path(feat_path)
        fear_t=fear_t+1
        if pred==2:
            fear=fear+1
    

    # Load the happy features
    hap=0
    hap_t=0
    for feat_path in glob.glob(os.path.join("feat_3","*.feat")):
        pred=load_path(feat_path)
        hap_t=hap_t+1
        if pred==3:
            hap=hap+1
    

    # Load the Sad features
    sad=0
    sad_t=0
    for feat_path in glob.glob(os.path.join("feat_4","*.feat")):
        pred=load_path(feat_path)
        sad_t=sad_t+1
        if pred==4:
            sad=sad+1
    

    # Load the Surprise features
    sur=0
    sur_t=0
    for feat_path in glob.glob(os.path.join("feat_5","*.feat")):
        pred=load_path(feat_path)
        sur_t=sur_t+1
        if pred==5:
            sur=sur+1
    
    # Load the neutral features
    neu=0
    neu_t=0
    for feat_path in glob.glob(os.path.join("feat_6","*.feat")):
        pred=load_path(feat_path)
        neu_t=neu_t+1
        if pred==6:
            neu=neu+1
    
    print "Successfully tested angry faces with accuracy = ",
    print angry*100.0/angry_total
    print angry_total
    print ".."
    
    print "Successfully tested Disgust faces with accuracy = ",
    print dis*100.0/dis_t
    print dis_t
    print ".." 

    print "Successfully tested Fear faces with accuracy = ",
    print fear*100.0/fear_t
    print fear_t
    print ".."

    print "Successfully tested Happy faces with accuracy = ",
    print hap*100.0/hap_t
    print hap_t
    print ".."

    print "Successfully tested Sad faces with accuracy = ",
    print sad*100.0/sad_t
    print sad_t
    print ".."

    print "Successfully tested Surprise faces with accuracy = ",
    print sur*100.0/sur_t
    print sur_t
    print ".."

    print "Successfully tested Neutral faces with accuracy = ",
    print neu*100.0/neu_t
    print neu_t
    print ".."
    total=angry_total+dis_t+sad_t+hap_t+fear_t+sur_t+neu_t
    pos_r=angry+dis+sad+hap+fear+sur+neu
    print "Total Accuracy of classifier is :",
    print pos_r*100.0/total
    print total
    print pos_r

