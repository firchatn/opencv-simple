import numpy as np
import cv2
import os

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

def write_data():
    id= raw_input('put id')
    nb = 0 
    while 1:
        ret, img = cap.read()
        only_face = np.array(10)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            nb +=1
            if nb ==1:
                os.system('mkdir user{0}'.format(id))
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            only_face = gray[y:y+h,x:x+w]
            cv2.imwrite("data/user."+str(id)+".jpg", only_face)
        cv2.imshow('live video',img)
        
        cv2.waitKey(100)
        if nb == 20:
            break
            cap.release()
            cv2.destroyAllWindows()

def train_data():
    pass


def recon_data():
    pass

