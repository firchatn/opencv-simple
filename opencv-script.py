import numpy as np
import cv2

#load img
img = cv2.imread("images/img.jpg",0)

# display img
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()


