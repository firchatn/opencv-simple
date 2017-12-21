import numpy as np
import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('images/img3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
only_face = np.array(10)
nb = 0 
for (x,y,w,h) in faces:
    nb +=1
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    only_face = img[y:y+h,x:x+w]
    #cv2.putText(img, 'firas', (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    cv2.imshow('face'+str(nb), only_face)
    
cv2.imshow('img',img)
cv2.imwrite("data/user"+".jpg", only_face)


#cv2.face.LBPHFisherFaceRecognizer()
#face_recognizer.train(faces, ['firas','inconnu'])


cv2.waitKey(0)
cv2.destroyAllWindows()
