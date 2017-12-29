import numpy as np
import cv2
import os
import os.path
from PIL import Image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.createLBPHFaceRecognizer()
cap = cv2.VideoCapture(0)

def write_data():
    id= int(input('put id'))
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

def train_data():
    images = []
    labels =[]
    dirs = os.listdir('./data')
    
    for dir in dirs:
        for i in range(20):
            face = cv2.imread('data/{0}/{1}.jpg'.format(dir,i+1))
            image_pil = Image.open('data/{0}/{1}.jpg'.format(dir,i+1)).convert('L')
            image = np.array(image_pil, 'uint8')
            nbr = int(dir[4:])
            print(nbr)
            faces = face_cascade.detectMultiScale(image)
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
                cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
                cv2.waitKey(10)

    face_recognizer.train(images, np.array(labels))
    face_recognizer.save('trainer/trainer.yml')
    cv2.destroyAllWindows()
            

def recon_data():
    while True:
        ret, im =cap.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        for(x,y,w,h) in faces:
            print("oki3")
            id_user, conf = face_recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
            if(id_user==1):
                 id_user='Firas'
            elif(id_user==2):
                 id_user='Moez'
            
            cv2.putText(im,str(id_user), (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
            cv2.imshow('im',im)
            cv2.waitKey(10)

