import numpy as np
import cv2
import os
import os.path
from PIL import Image
import json
import pandas as pd
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer = cv2.face.createLBPHFaceRecognizer()
cap = cv2.VideoCapture(0)

def doesFileExists(filePathAndName):
    return os.path.exists(filePathAndName)
  

if doesFileExists('userinfo/sdata.json'):
    json_data=open("userinfo/sdata.json")
    statistics = json.load(json_data)
else:
    statistics = {"Firas": 1, "Moez": 2, "Med Ali": 3, "Moez 2": 4, "Moez 3": 4}
    with open('userinfo/sdata.json', 'w') as outfile:
        json.dump(statistics, outfile)

if doesFileExists('userinfo/user.json'):
    json_data=open("userinfo/user.json")
    docs = json.load(json_data)
else:
    docs = { 1 : 'Firas' , 2 : 'Moez' , 3 : 'Med Ali' , 4 : 'Moez 2' ,
     5 : 'Moez 3' }
    with open('userinfo/user.json', 'w') as outfile:
        json.dump(docs, outfile)




def count_data_in(name):
    found  = False
    json_data=open("userinfo/sdata.json")
    jdata = json.load(json_data)
    for key, value in jdata.items():
        if name == key:
            jdata[key] += 1
            found = True
    if not found:
        jdata[name] = 1

    with open("userinfo/sdata.json", "w") as jsonFile:
        json.dump(jdata, jsonFile)
            

def courbe_day():
    listName = []
    listValue = []
    json_data=open("userinfo/sdata.json")
    jdata = json.load(json_data)

    for key, value in jdata.items():
        if value in listValue:
            value+=1
        listName.append(key)
        listValue.append(value)
    x = np.array(listValue)
    y = np.array( range(1,len(listName)*2,2))
    my_xticks = listName
    plt.xticks(x, my_xticks)
    plt.plot(x, y//2)
    plt.show()
    

def write_data_static():
    id = len(os.listdir('./data')) + 1
    nb = 0 
    while nb < 20:
        img = cv2.imread('../images/ali.jpg')
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
            
    while 1 :
        choice = input('you know is person : yes/no')
        if choice == 'yes' or choice == 'no':
            break
    if choice == 'yes':
        ch = input('name this person')
    else:
        ch = 'inconnu'
    with open("userinfo/user.json", "r") as jsonFile:
        docs = json.load(jsonFile)
        
    docs[id] = ch

    with open("userinfo/user.json", "w") as jsonFile:
        json.dump(docs, jsonFile)


def write_data():
    id = len(os.listdir('./data')) + 1 
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
            cap.release()
            cv2.destroyAllWindows()
            break
            
    while 1 :
        choice = input('you know is person : yes/no')
        if choice == 'yes' or choice == 'no':
            break
    if choice == 'yes':
        ch = input('name this person')
    else:
        ch = 'inconnu'
    with open("userinfo/user.json", "r") as jsonFile:
        docs = json.load(jsonFile)
        
    docs[id] = ch

    with open("userinfo/user.json", "w") as jsonFile:
        json.dump(docs, jsonFile)


def train_data():
    images = []
    labels =[]
    dirs = os.listdir('./data')
    for dir in dirs:
        nbimage = len(os.listdir('./data/{0}'.format(dir)))
        for i in range(nbimage):
            face = cv2.imread('data/{0}/{1}.jpg'.format(dir,i+1))
            image_pil = Image.open('data/{0}/{1}.jpg'.format(dir,i+1)).convert('L')
            image = np.array(image_pil, 'uint8')
            nbr = int(dir[4:])
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
            id_user, conf = face_recognizer.predict(gray[y:y+h,x:x+w])
            cv2.rectangle(im,(x-10,y-10),(x+w+10,y+h+10),(225,255,255),2)
            name = docs[str(id_user)]
            count_data_in(name)
            cv2.putText(im,str(name), (x,y-15),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, 25)
            cv2.imshow('im',im)
            cv2.waitKey(10)

