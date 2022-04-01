import cv2 as c 
import numpy as n 
from os import listdir
from os.path import isfile , join

# get the training data we previously made 
data_path= './faces/user/'
onlyfiles=[f for f in listdir(data_path) if isfile(join(data_path, f))]

#create a numpy array for training data and labels
training_data, labels=[], []

#open training images in our datapath
#create a numpy array for training data
for i , files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images=c.imread(image_path, c.IMREAD_GRAYSCALE)
    training_data.append(n.asarray(images,dtype=n.uint8))
    labels.append(i)
    
#create a numpy array for both training data and labels 
labels=n.asarray(labels,dtype=n.int32)

#initialize facial recognizer 
model= c.createLBPHFaceRecognizer()
# note : for openCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# and it was my wrong in the other codes for face recognition

# let's train our model 
model.train(n.asarray(training_data),n.asarray(labels))
print 'model trained successfully'