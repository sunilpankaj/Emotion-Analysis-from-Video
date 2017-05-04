import numpy as np
import cv2

#cap = cv2.VideoCapture(0)  #record from camera
vidcap = cv2.VideoCapture('vids/put.3gp')
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  cv2.imwrite("putin/frame%d.jpg" % count, image)     # save frame as JPEG file
  count += 1
print count

