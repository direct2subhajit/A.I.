#Importing necessary libraries
import cv2
import time
import imutils

#Loading the video
cam = cv2.VideoCapture(2)

#Timeout for it to look good
time.sleep(1)

#We initialize the first frame to empty
firstFrame=None
area = 500

#We go through all the frames
while True:
    #We get the frame
    _,img = cam.read()
    text1 = "Normal"
    img = imutils.resize(img, width=800)

    #We convert to greyscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #We apply smoothing to remove noise
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0)
    if firstFrame is None:
            firstFrame = gaussianImg
            continue

    #Calculation of the difference between the back ground and the curret frame
    imgDiff = cv2.absdiff(firstFrame, gaussianImg)

    #We apply a threshold
    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1]

    #We expand the threshold to cover holes
    threshImg = cv2.dilate(threshImg, None, iterations=2)

    #We copy the threshold to detect the contours, we look for contour in the image
    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #We go through all the contours found
    for c in cnts:

            #We remove the smallest contours
            if cv2.contourArea(c) < area:
                    continue

            #We obtain the bunds of the contour, the largest rectangle that encompasses the contour
            (x, y, w, h) = cv2.boundingRect(c)

            #We draw the rectangle of the bounds
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            count= str(len(cnts))
            text = "Moving Object detected " + "Count: "+ count
    print(text)
    cv2.putText(img, text, (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("cameraFeed",img)

    #We capture the key to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

#We release the camera and close all the windows
cam.release()
cv2.destroyAllWindows()
