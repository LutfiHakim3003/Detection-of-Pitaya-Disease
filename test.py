import numpy as np
import os
import pandas as pd
import cv2
from skimage.feature import greycomatrix,greycoprops
from skimage.measure import label
import skimage
import matplotlib.pyplot as plt

INPUT_SCAN_FOLDER="D:/KULIAH/PROJECT TA/Pitaya/pitaya/test/"
image_folder_list = os.listdir(INPUT_SCAN_FOLDER)
proList = ['contrast', 'dissimilarity']
featlist = ['X1','X2','X3','X4','X5','X6','X7', 'X8','Y']
properties =np.zeros(2)
glcmMatrix = []
final=[]

print ("sedang proses membaca citra dan proses segmentasi.......")

for i in range(len(image_folder_list)):

        img =cv2.imread(INPUT_SCAN_FOLDER+image_folder_list[i])
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        c1 = lab[:, :, 0]
        c2 = lab[:, :, 1]
        c3 = lab[:, :, 2]

        low = np.array([30])
        up = np.array([126])
        mask = cv2.inRange(c2, low, up)
        img[mask>0]=(255,255,255)

        imgplot = plt.imshow(img)
        plt.show()

        red_channel = img[:,:,0]
        green_channel = img[:,:,1]
        blue_channel = img[:,:,2]
        blue_channel[blue_channel == 255] = 0
        green_channel[green_channel == 255] = 0
        red_channel[red_channel == 255] = 0
        
        red_mean = np.mean(red_channel)
        green_mean = np.mean(green_channel)
        blue_mean = np.mean(blue_channel)
        
        red_std = np.std(red_channel)
        green_std = np.std(green_channel)
        blue_std = np.std(blue_channel)

        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # images = images.f.arr_0


        glcmMatrix = (greycomatrix(gray_image, [1], [0], levels=2 ** 8))
        for j in range(0, len(proList)):
            properties[j] = (greycoprops(glcmMatrix, prop=proList[j]))

        features = np.array(
            [ red_std, green_std, blue_std,red_mean,green_mean,blue_mean,properties[0], properties[1],'?'])
        final.append(features)
        
df = pd.DataFrame(final, columns=featlist)
filepath =  "Test.csv"
df.to_csv(filepath)

import csv
Test = []
with open('Test.csv')as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for row in csv_reader:
                Test.append(row)
labels = Test.pop(0)

print ("hasil ekstraksi dari citra ")
print(labels)
print(Test)

from classifier import *

if pred == 0:
        cv2.namedWindow("img", cv2.WINDOW_NORMAL)
        cv2.putText(img, "Terkena Penyakit Busuk Batang", (150,200), cv2.FONT_HERSHEY_SIMPLEX,5,(255,255,2555),20)
        cv2.imshow("img",img)
        cv2.imwrite("img.jpg", img)
        cv2.waitKey(0)
elif pred == 1:
	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.putText(img, "Batang Terkena Penyakit Cacar", (150,200), cv2.FONT_HERSHEY_SIMPLEX,5,(255, 255, 255),20)
	cv2.imshow("img",img)
	cv2.imwrite("img.jpg", img)
	cv2.waitKey(0)
elif pred == 2:
	cv2.namedWindow("img", cv2.WINDOW_NORMAL)
	cv2.putText(img, "Batang Tersengat Serangga", (150,200), cv2.FONT_HERSHEY_SIMPLEX,5,(255, 255, 255),20)
	cv2.imshow("img",img)
	cv2.imwrite("img.jpg", img)
	cv2.waitKey(0)
