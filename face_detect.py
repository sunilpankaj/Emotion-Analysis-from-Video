import dlib
import Image
from skimage import io
import matplotlib.pyplot as plt
import sys
import glob
import cv2
import numpy as np

def detect_faces(image):

    # Create a face detector
    face_detector = dlib.get_frontal_face_detector()

    # Run detector and get bounding boxes of the faces on image.
    detected_faces = face_detector(image, 1)
    face_frames = [(x.left(), x.top(),
                    x.right(), x.bottom()) for x in detected_faces]

    return face_frames

# Read the resized_image
cv_img = []
for img in glob.glob("hilar/*.jpg"):
    n= cv2.imread(img,0)
    cv_img.append(n)
# Detect faces
count = 0
for image in cv_img:
	detected_faces = detect_faces(image)
	for n, face_rect in enumerate(detected_faces):
		face1 = Image.fromarray(image).crop(face_rect)
		face = np.array(face1)
		resized_image = cv2.resize(face, (48, 48))
		#print resized_image.shape
		face_img = cv2.imwrite("hilary_face_images/face%d.jpg" % count, resized_image)
		count += 1


                                        
