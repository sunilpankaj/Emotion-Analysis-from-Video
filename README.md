# Emotion-Analysis-in-Video- Using CNN


Files:-
1. record_video.py  =  For recording video using webcam
2. getimages.py  =  For breaking the videos into frames
3. face_detect.py = For detecting faces from frames extracted from videos
4. make_csv.py  = Creating raw data from detected faces to fetch to the CNN model
5. classifier.py = SVM classifier (Just for comparing the accuracy)
6. plot.py  = plotting results

folders:-
1. obama =  Image frames of of obama and Trump extracted from the video
2. obama_face_images = Detected and reshaped face images of Obama and Trump from the extracted faces
3. train_complete = CNN model testing with YouTube videos
4. train  =  training and testing CNN with fer2013

5. saved_model = This folder contains Saved CNN model which can be imported anywhere for testing purpose. Because of the huge size(around 0.8 GB) of Tesorflow CNN model we are not adding anything to this folder. When you run the "train.py" file, a CNN model will be autometically stored in this folder which can be used later anywhere.

Links:-
1. fer2013 dataset http://www-etud.iro.umontreal.ca/~goodfeli/fer2013.html

Steps to run:-
1. To record the video from webcam run record_video.py
2. To break the video run getimages.py
3. For face detection run face_detect.py
4. To convert faces to raw data run make_csv.py
5. To train the CNN run train.py
6. for testing on fer2013 dataset run test.py
7. for testing on manual video run test_obama.py



