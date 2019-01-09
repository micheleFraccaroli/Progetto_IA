import os
import random as ra
import numpy as np
import PIL
from PIL import Image
from keras.preprocessing import image

class Load_Module:
	def loading(self):
		PATH = os.getcwd()
		folder_path = PATH + "/DigitDataset/Images/Test/"
		labeling_list = os.listdir(folder_path)

		data = []

		# for im in img_list:
		# 	img_path = data_path + im
		# 	loaded = image.load_img(img_path)
		# 	scalesize = (28,28)
		# 	x_scaled = loaded.resize(scalesize, PIL.Image.ANTIALIAS)
		# 	x = np.array(x_scaled)
		# 	data.append(x)

		for lb in labeling_list:
			i = ra.randint(1,9)
			data_path = folder_path + "/" + str(i) + "/"
			for j in range(2):
				img_list = os.listdir(data_path)
				num_img = ra.randint(1,len(img_list)-1)
				img_path = data_path + img_list[num_img]
				loaded = image.load_img(img_path)
				scalesize = (28,28)
				x_scaled = loaded.resize(scalesize, PIL.Image.ANTIALIAS)
				x = np.array(x_scaled)
				data.append(x)

		data = np.array(data)
		
		return data

if __name__ == '__main__':
	l = Load_Module()
	l.loading()