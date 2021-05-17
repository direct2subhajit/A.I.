#Importing necessary libraries
import cv2, os
haar_file = 'haarcascade_frontalface_default.xml'

#Creating database folder
datasets = 'dataset'  
sub_data = 'champ'     

#Creating path
path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

#Creating face cascade frame
(width, height) = (130, 100)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_file)

#Loading the video
webcam = cv2.VideoCapture(2)

#We go through all the frames
while True:

    #We get the frame
    (_, im) = webcam.read()
    text="No Persion Detected"

    #We convert to greyscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #Detect Multi Scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    for (x,y,w,h) in faces:

        #Draw rectangle on face co-ordinates
        cv2.rectangle(im,(x,y),(x+w,y+h),(255,0,0),2)
        print (faces.shape)
        text= "Person Detected"+":"+str(faces.shape[0])
        print(text)
    ImgWithText = cv2.putText(im,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
        

    #We capture the key to exit
    cv2.imshow('FaceDetection', im)
    key = cv2.waitKey(10)
    if key == 27:
        break

#We release the camera and close all the windows
webcam.release()
cv2.destroyAllWindows()
