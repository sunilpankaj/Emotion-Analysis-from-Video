import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigs
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn import svm
#import cv2

#np.set_printoptions(threshold='nan')
#read = cv2.imread("face_images/face1.jpg",0)
#re = cv2.resize(read, (2304,1))
#re = pd.read_csv("output.csv")
#print re.shape
df = pd.read_csv('fer2013.csv')
#df = df.head(10)
df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype=np.float))
def make_image_vector(image_list, new_size):
	#print "image list", image_list
	D = np.empty([len(image_list), new_size])
	i = 0
	for image in image_list:
		D[i, :] = image
		i = i + 1
	return D

X = make_image_vector(df['pixels'], 48*48)
#r = make_image_vector(re, 48*48)
#print "r = ", re
y = df['emotion']
#print X.shape
#print y.shape

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print "size of train data = ", X_train.shape,"size of test data = ", X_test.shape
print "size of target train data = ", y_train.shape, "size of target test data = ",y_test.shape

clf = svm.SVC()
cl = clf.fit(X_train, y_train)
y_pred = cl.predict(X_test)
#y_pred = cl.predict(test)
#print y_pred
score = accuracy_score(y_test, y_pred)
print score	
