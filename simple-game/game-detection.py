import numpy as np
import cv2
import random 
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(1)
score = 100
imgf = cv2.imread('1.png')
resf = cv2.resize(imgf,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_CUBIC)
t = 0 # time
def too_passed(oldepoch):
    return time.time() - oldepoch >= 2
while 1 and score >= 0:
    ret, img = cap.read()
    only_face = np.array(10)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,250,250),2)
        only_face = gray[y:y+h,x:x+w]

        
        rows,cols,channels = resf.shape
        rows1,cols1,channels1 = img.shape
        print(rows1)
        print(cols1)
        x1 = random.randint(1,rows1)
        x2 = random.randint(1,rows1)
        y1 = random.randint(1,cols1)
        y2 = random.randint(1,cols1)
        roi = img[0:rows, 0:cols ]
        img2gray = cv2.cvtColor(resf,cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 0, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
        img2_fg = cv2.bitwise_and(resf,resf,mask = mask)
        dst = cv2.add(img1_bg,img2_fg)
        img[0:rows, 0:cols ] = dst

        
        if too_passed(t):
            print("ok")
            
            try:
                a = img[x1:x1+resf.shape[0], x1:x1+resf.shape[1]] = dst
                t = time.time()
            except Exception:
                pass
            
            
        if x1 in only_face  :
            print("ok")
            #score -=10 
    cv2.putText(img,"Score"+str(score), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

   
    cv2.imshow('live video',img)
    
    cv2.waitKey(30) & 0xff
    

cap.release()
cv2.destroyAllWindows()
