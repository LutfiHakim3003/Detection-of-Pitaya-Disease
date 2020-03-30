import cv2
import numpy as np

image = cv2.imread("image3.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_blue = np.array([36, 25, 25])
upper_blue = np.array([86, 255, 255])

mask = cv2.inRange(hsv, lower_blue, upper_blue)

image[mask>0]=(255, 255, 255)

cv2.imwrite('image7.jpg',image)
cv2.waitKey(0)
