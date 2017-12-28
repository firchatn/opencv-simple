import cv2
import numpy as np
# Reading Image
img = cv2.imread("../images/im1.png")

cv2.namedWindow("Image Show",cv2.WINDOW_NORMAL)

cv2.imshow("Image Show",img)

cv2.waitKey() 
