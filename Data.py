import cv2 as c 
import numpy as n 

#load HAAR face classifier 
face_classifier=c.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load functions
def face_extractor(img):
    #function detects faces and return the cropped face
    #if no faces detected it returns the input image 
    
    gray=c.cvtColor(img,c.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    
    if faces is ():
        return None
    
    #crop all faces found 
    for (x,y,w,h) in faces :
        cropped_face=img[y:y+h , x:x+w]
    
    return cropped_face
# initialize webcam
cap=c.VideoCapture(0)
count=0

#collect 100 samples of your face from webcam input
while True :
    ret,frame =cap.read()
    if face_extractor(frame) is not None :
        count +=1
        face= c.resize(face_extractor(frame),(200,200))
        face=c.cvtColor(face,c.COLOR_BGR2GRAY)
        
        #save file in specified directory with unique name
        file_name_path='./faces/user/' + str(count)+'.jpg'
        c.imwrite(file_name_path, face)
        
        #put count on images and display live count 
        c.putText(face, str(count),(50,50),c.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
        c.imshow ('face cropper', face)
    else :
        print 'face not found'
        pass
    if c.waitKey(1)==13 or count ==100 :
        # 13 is the enter key
        break
        
cap.release()
c.destroyAllWindows()
print ('collecting samples complete')
        
        