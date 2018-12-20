import os
import numpy as np
import PIL
from PIL import Image
from keras.preprocessing import image

class Load_Module:

	def loading(self):
		PATH = os.getcwd()
		data_path = PATH + "/DigitDataset/Images/"

		img_list = os.listdir(data_path)

		data = []

		for im in img_list:
			img_path = data_path + im
			loaded = image.load_img(img_path)
			scalesize = (28,28)
			x_scaled = loaded.resize(scalesize, PIL.Image.ANTIALIAS)
			x = np.array(x_scaled)
			data.append(x)

		print("DATA TYPE  → " + str(type(data)))
		data = np.array(data)
		print("DATA SHAPE → " + str(data.shape))
		
		return data