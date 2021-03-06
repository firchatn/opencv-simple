import numpy as np
import cv2
import os
import os.path
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture('video/outpy.avi')

def save_detect_faces():
    id= int(input('video number : '))
    nb = 0 
    while 1:
        ret, img = cap.read()
        only_face = np.array(10)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            nb +=1
            if nb ==1:
                os.system('mkdir data/user{0}'.format(id))
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            only_face = gray[y:y+h,x:x+w]
            cv2.imwrite("data/user"+str(id)+"/"+str(nb)+".jpg", only_face)
        cv2.imshow('live video',img)
        cv2.waitKey(1)
        if nb == 20:
            break
            cap.release()
            cv2.destroyAllWindows()
