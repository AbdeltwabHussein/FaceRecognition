import cv2 as c
import numpy as n 

face_classifier= c.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_detector(img, size=0.5):
    #convert image to grayscale
    gray=c.cvtColor(img,c.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is ():
        return img, []
    
    for (x,y,w,h) in faces :
        c.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi=img[y:y+h, x:x+w]
        roi=c.resize(roi,(200,200))
    return img, roi

#open webcam
cap=c.VideoCapture(0)

while True :
     ret, frame = cap.read()
     image, face= face_detector(frame)
     
     try :
        face=c.cvtColor(face, c.COLOR_BGR2GRAY)
        #pass face to prediction model 
        #'results' comprises f a tuple containing the label and the confidence value
        results = model.predict(face)
        
        if results[1]<500:
            confidence=int(100*(1-(results[1])/300) )
            display_string=str(confidence)+ '% confidence it is user'
        
        #c.putText(image, display_string, (100,120), c.FONT_HERSHEY_COMPLEX,1,(255,120,150),2)
        
        if confidence >78 :
            c.putText(image,"Welcome,Have a nice time",(120,120),c.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            #c.putText(image,"Your Fingerprint Please",(120,120),c.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)
            c.putText(image,'unlocked',(250,450),c.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            c.imshow('Face Cropper', image )
        elif 0 < confidence < 65  :
            c.putText(image,"U r not the Driver",(150,90),c.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)
            c.putText(image,"Your Fingerprint Please",(120,120),c.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)
            c.putText(image,'locked',(250,450),c.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)
            c.imshow('Face Cropper',image)
        else:
            c.putText(image,'locked',(250,450),c.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            c.imshow('Face Cropper', image )
            
     except:
        c.putText(image,'No face found',(200,120),c.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)
        c.putText(image,'locked',(250,450),c.FONT_HERSHEY_COMPLEX,1,(0,0,250),2)
        c.imshow('Face Cropper',image)
        pass
     if c.waitKey(1)==13: # 13 is the enter key
        break

cap.release()
c.destroyAllWindows()
        
        