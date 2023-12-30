import cv2 as cv
import numpy as np

face = cv.CascadeClassifier('D:\\OpenCV\\[P9] Face and Eye Detection\\haarcascade_frontalface_default.xml')
eye = cv.CascadeClassifier('D:\\OpenCV\\[P9] Face and Eye Detection\\haarcascade_eye.xml') #for detecting eyes

def detection(img):
    gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,1.3,5)
    
    for(x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(127,0,125),3)
        
        roi_gray = gray[y:y+h, x:x+w]
        
        roi_color = img[y:y+h, x:x+w]
        
        eyes = eye.detectMultiScale(roi_gray,1.3,3)
        for (ex,ey,ew,eh) in eyes:
            cv.circle(roi_color,(ex+27,ey+27),20,(255,255,0),2)

    return img

cap = cv.VideoCapture(0)
# incresing brightness of camera
cap.set(10,200)

while True:
    if cap.isOpened():
        ret, frame = cap.read()
        frame = cv.flip(frame,2)
        frame =cv.resize(frame,(500,500))
        cv.imshow('org',detection(frame))
        key =cv.waitKey(25)
        if key == 27:
            break

cap.release()
cv.destroyAllWindows()