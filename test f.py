import numpy as np
import csv
import os
from cv2 import imread, imwrite
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

for i in range(len(image_folder_list)):

        img =cv2.imread(INPUT_SCAN_FOLDER+image_folder_list[i])


        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        c1 = lab[:, :, 0]
        c2 = lab[:, :, 1]
        c3 = lab[:, :, 2]

        low = np.array([30])
        up = np.array([126])

        mask = cv2.inRange(c2, low, up)

        img[mask>0]=(255, 255, 255)
        
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

csv = pd.read_csv('Test.csv')
csv.plot.bar(y=['X1','X2','X3','X4','X5','X6','X7','X8'],subplots=True, layout=(1,8))
plt.show()


#KNN
# Menghitung jarak perkiraan dari dua buah titik
def hitung_perkiraan(x, y):
	return abs(x['X1'] - y['X1']) + abs(x['X2'] - y['X2'])+ abs(x['X3'] - y['X3']) + abs(x['X4'] - y['X4'])+ abs(x['X5'] - y['X5']) + abs(x['X6'] - y['X6'])+ abs(x['X7'] - y['X7']) + abs(x['X8'] - y['X8'])


# Memprediksi data dari datasets
def prediksi_data(nilai, data, x):
	daftar_perkiraan = [{'hitung_perkiraan': float('inf')}]
	for dataset in data:
		hasil = hitung_perkiraan(nilai, dataset)
		if hasil < daftar_perkiraan[-1]['hitung_perkiraan']:
			if len(daftar_perkiraan) >= x:
				daftar_perkiraan.pop()
			i = 0
			while i < len(daftar_perkiraan)-1 and hasil >= daftar_perkiraan[i]['hitung_perkiraan']:
				i += 1
			daftar_perkiraan.insert(i, {'hitung_perkiraan': hasil, 'Y': dataset['Y']})
	daftar_nilai = list(map(lambda x: x['Y'], daftar_perkiraan))
	return max(daftar_nilai, key=daftar_nilai.count)


# Klasifikasi datatest berdasarkan data pada file DataTrain
def hasil_klasifikasi(data_test, data_train, k):
	for d_test in data_test:
		d_test['Y'] = prediksi_data(d_test, data_train, k)
		if d_test['Y'] == 0:
			cv2.namedWindow("img", cv2.WINDOW_NORMAL)
			cv2.putText(img, "Terkena Penyakit Busuk Batang", (150,200), cv2.FONT_HERSHEY_SIMPLEX,5,(80, 255, 85),20)
			cv2.imshow("img",img)
			cv2.imwrite("img.jpg", img)
			cv2.waitKey(0)
		elif d_test['Y']== 1:
			cv2.namedWindow("img", cv2.WINDOW_NORMAL)
			cv2.putText(img, "Batang Terkena Penyakit Cacar", (150,200), cv2.FONT_HERSHEY_SIMPLEX,5,(80, 255, 85),20)
			cv2.imshow("img",img)
			cv2.imwrite("img.jpg", img)
			cv2.waitKey(0)
		elif d_test['Y'] == 2:
			cv2.namedWindow("img", cv2.WINDOW_NORMAL)
			cv2.putText(img, "Batang Tersengat Serangga", (150,200), cv2.FONT_HERSHEY_SIMPLEX,5,(80, 255, 85),20)
			cv2.imshow("img",img)
			cv2.imwrite("img.jpg", img)
			cv2.waitKey(0)

# Fungsi untuk membaca data dari file csv 
def baca_input_csv(f, kondisi=False):
	dataset = [] # buat array kosong untuk menampung nilai dari file csv yang dibaca
	with open(f) as csv_input:
		baca_csv = csv.DictReader(csv_input, skipinitialspace=True)
		for baris in baca_csv:
			dataset.append({'i': int(baris['']), 'X1': float(baris['X1']), 'X2': float(baris['X2']), 'X3': float(baris['X3']), 'X4': float(baris['X4']), 'X5': float(baris['X5']),'X6': float(baris['X6']),'X7': float(baris['X7']),'X8': float(baris['X8']),'Y': int(baris['Y']) if kondisi else baris['Y']}) 
	return dataset


# Main program untuk menjalankan fungsi yang sudah dibuat sebelumnya
if __name__ == '__main__':
	hasil_klasifikasi(baca_input_csv('Test.csv'), baca_input_csv('Training.csv', kondisi=True), 4) # Nilai parameter k =4


