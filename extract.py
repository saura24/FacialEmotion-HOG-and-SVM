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
import macpath as mp
import numpy as np
#To Display
import matplotlib.pyplot as plt
from skimage import data, color, exposure


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

if __name__ == "__main__":
    des_type="HOG"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('0/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_0"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "Angry features saved in {}"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('1/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_1"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "Disgust features saved in {}"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('2/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_2"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "fear features saved in {}"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('3/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_3"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "Happy features saved in {}"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('4/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_4"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "Sad features saved in {}"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('5/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6, 6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_5"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "Surprise features saved in {}"
    print "Calculating the descriptors for the positive samples and saving them"
    for im_path in glob.glob('6/*'):
        im = imread(im_path, as_grey=True)
        im = exposure.adjust_gamma(im)
        fd, hog_image = hog(im, orientations=9, pixels_per_cell=(6,6),
                    cells_per_block=(3, 3), visualise=True)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = "feat_6"+"/"+fd_name
        joblib.dump(fd, fd_path)
    print "."
    print "Neutral features saved in {}"



    print "Completed calculating features from training images"