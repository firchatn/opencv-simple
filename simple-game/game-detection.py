import numpy as np
import cv2
from random import randrange

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
score = 100
imgf = cv2.imread('1.png')
resf = cv2.resize(imgf,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)

while 1:
    ret, img = cap.read()
    only_face = np.array(10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,250,250),2)
        only_face = gray[y:y+h,x:x+w]
        #img[y:y+h,x:x+w] = resf
        #dst = cv2.addWeighted(img,0.7,resf,0.3,0)


        rows,cols,channels = resf.shape
        roi = img[0:rows, 0:cols ]
        img2gray = cv2.cvtColor(resf,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(resf,resf,mask = mask)
        dst = cv2.add(img1_bg,img2_fg)
        img[0:rows, 0:cols ] = dst
        #img[2:rows, 2:cols ] = dst
        if dst in only_face:
            score -=10
        
    cv2.putText(img,"Score"+str(score), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

   
    cv2.imshow('live video',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
