# Import the required modules
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import glob
import numpy as np
import os
#Define path to the model
model_path='model_path/svm.model'
def load_feat(feat_path,label):
    fd = joblib.load(feat_path)
    fds.append(fd)
    labels.append(label)

if __name__ == "__main__":
    fds = []
    labels = []
    # Load the Angry features
    for feat_path in glob.glob(os.path.join("feat_0","*.feat")):
        load_feat(feat_path,0)
    print "Successfully loaded Angry features"
    # Load the Disgust features
    for feat_path in glob.glob(os.path.join("feat_1","*.feat")):
        load_feat(feat_path,1)
    # Load the fear features
    for feat_path in glob.glob(os.path.join("feat_2","*.feat")):
        load_feat(feat_path,2)
    # Load the Sad features
    for feat_path in glob.glob(os.path.join("feat_3","*.feat")):
        load_feat(feat_path,3)
    # Load the happy features
    for feat_path in glob.glob(os.path.join("feat_4","*.feat")):
        load_feat(feat_path,4)
    # Load the surprise features
    for feat_path in glob.glob(os.path.join("feat_5","*.feat")):
        load_feat(feat_path,5)
    # Load the Neutral features
    for feat_path in glob.glob(os.path.join("feat_6","*.feat")):
        load_feat(feat_path,6)
    
    print "Successfully loaded disgust features"
    clf = LinearSVC()
    print "Training a Linear SVM Classifier"
    clf.fit(fds, labels)
    # If feature directories don't exist, create them
    joblib.dump(clf, model_path)
    print "Classifier saved to {}".format(model_path)
