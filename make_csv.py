import Image
from PIL import Image
import numpy as np
import csv
import glob
import cv2
import pandas as pd 
import string
import pyexcel as pe
#np.set_printoptions(threshold='nan') 
# Read the image
cv_img = []
for img in glob.glob("face_images/*.jpg"):
    im= cv2.imread(img,0)
    n = im.shape[0]
    m = im.shape[1]
    resized_image = cv2.resize(im, (n*m,1))
    #print resized_image.shape
    #resized_image = res_image.tolist()
    cv_img.append(np.array(resized_image).tolist())
#print cv_img

new_list = []
'''for i in cv_img:
    print len(i[0])'''

with open("putin.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    for i in cv_img:
    	wr.writerow(i[0])

df = pd.read_csv("putin.csv")
print df.shape

#########################################
'''cv_img = []
for img in glob.glob("hilary_face_images/*.jpg"):
    im= cv2.imread(img,0)
    n = im.shape[0]
    m = im.shape[1]
    resized_image = cv2.resize(im, (n*m,1))
    #print resized_image.shape
    #resized_image = res_image.tolist()
    cv_img.append(np.array(resized_image).tolist())
#print cv_img

new_list = []
for i in cv_img:
    print len(i[0])

with open("hilary.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    for i in cv_img:
        wr.writerow(i[0])

df = pd.read_csv("putin.csv")
print df.shape

######################################################
cv_img = []
for img in glob.glob("obama_face_images/*.jpg"):
    im= cv2.imread(img,0)
    n = im.shape[0]
    m = im.shape[1]
    resized_image = cv2.resize(im, (n*m,1))
    #print resized_image.shape
    #resized_image = res_image.tolist()
    cv_img.append(np.array(resized_image).tolist())
#print cv_img

new_list = []
for i in cv_img:
    print len(i[0])

with open("obama.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    for i in cv_img:
        wr.writerow(i[0])

df = pd.read_csv("obama.csv")
print df.shape'''