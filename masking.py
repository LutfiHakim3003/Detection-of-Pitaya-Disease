import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img1 = cv.imread("s2.jpg")

cvt_image = cv.cvtColor(img1,cv.COLOR_BGR2HSV_FULL)

c1 = cvt_image[:, :, 0]
c2 = cvt_image[:, :, 1]
c3 = cvt_image[:, :, 2]


ret,thresh_L = cv.threshold(c1,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
ret,thresh_a = cv.threshold(c2,100,150,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
ret,thresh_b = cv.threshold(c3,0.7,1,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)


f, (ax1, ax2, ax3,ax4) = plt.subplots(1,4, figsize=(10,5))
ax1.set_title("LAB")
ax1.imshow(cvt_image)
ax2.set_title("H")
ax2.imshow(thresh_L)
ax3.set_title("S")
ax3.imshow(thresh_a)
ax4.set_title("V")
ax4.imshow(thresh_b)
plt.show()

