import numpy as np
import cv2
from random import randrange
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)

id= randrange(0, 101, 2)
while 1:
    ret, img = cap.read()
    #img2 = np.zeros((150,150,3), np.uint8)
    only_face = np.array(10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,250,250),2)
        only_face = gray[y:y+h,x:x+w]
        cv2.imwrite("data/user."+str(id)+".jpg", only_face)
        
        cv2.putText(img,"Firas", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

   
    cv2.imshow('live video',img)
    cv2.imshow('face', only_face)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
