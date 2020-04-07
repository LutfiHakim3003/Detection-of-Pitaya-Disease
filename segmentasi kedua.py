import cv2
import numpy as np

image = cv2.imread("c2.jpg")

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

c1 = lab[:, :, 0]
c2 = lab[:, :, 1]
c3 = lab[:, :, 2]

low = np.array([30])
up = np.array([126])

mask = cv2.inRange(c2, low, up)

image[mask>0]=(255, 255, 255)

cv2.imwrite('h7.jpg',image)
cv2.waitKey(0)
