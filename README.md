# FacialEmotion-HOG-and-SVM

This is a python code to classify the 7 universal emotions which are Angry, Disgust, Fear, Sad, Happy, Surprise, Neutral.
fer2013.csv contains all the files. To read more refer this website:https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

Modules required:
1. skimage
2. cv2

To Run this program follow the given steps:
1. Run the "gen_record.py" using
      python gen_record.py
2. Now 3 folders are created.In each folder create 7 folder with names 'feat_0', 'feat_1', 'feat_2', 'feat_3', 'feat_4', 'feat_5', 'feat_6'.
3. In each folder copy the extract.py , test-classifier.py.Copy train-classifier.py and "model" folder in training folder.
4. Run extract.py in each folder.This will save all the features in feature folders("feat_0",etc)
5. In training folder run train-classifier.py.This will save the svm models in model folder.Copy this model folder to test folders. 
6. No run the test-classifier.py in the test folder.

This dataset is taken from 
"Challenges in Representation Learning: A report on three machine learning
contests." I Goodfellow, D Erhan, PL Carrier, A Courville, M Mirza, B
Hamner, W Cukierski, Y Tang, DH Lee, Y Zhou, C Ramaiah, F Feng, R Li,
X Wang, D Athanasakis, J Shawe-Taylor, M Milakov, J Park, R Ionescu,
M Popescu, C Grozea, J Bergstra, J Xie, L Romaszko, B Xu, Z Chuang, and
Y. Bengio. arXiv 2013.

See fer2013.bib for a bibtex entry.
