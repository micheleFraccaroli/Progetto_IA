from time import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
from scipy.misc import toimage
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras import backend as K
from Load_Module import Load_Module

class useNN:

	def show_imgs(self,X,lab):
	    pyplot.figure(1)
	    k = 0
	    for i in range(0,4):
	        for j in range(0,4):
	            pyplot.subplot2grid((4,4),(i,j))
	            pyplot.imshow(toimage(np.squeeze(X[k],2)))
	            pyplot.annotate(lab[k],xy=(0,0), xytext=(.8,-2), fontsize=10, fontweight='bold', color='r')
	            k = k+1
	    # show the plot
	    pyplot.subplots_adjust(hspace=1, wspace=1)
	    pyplot.show()

	def fire(self):
		# data = np.load('mnist.npz')
		# X = data['x_train']
		# Y = data['y_train']
		# X_test = data['x_test']
		# Y_test = data['y_test']

		lm = Load_Module()
		X_test = lm.loading()

		X_test = X_test.reshape(X_test.shape[0],28, 28,1)

		json_file = open('mnist_model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		model.load_weights('mnist.h5')

		labels = ['0','1','2','3','4','5','6','7','8','9','N']
		 
		indices = np.argmax(model.predict(X_test[:16]),1)
		predictions = model.predict(X_test[:16])
		print("MODEL PREDICTIONS -------------------------------------\n " + str(predictions))
		print("\n-----------------------------------------------------\n")

		k = 0
		for i in range(len(predictions)):
			for j in predictions[i]:
				if(j > 0):
					if(k == 0):
						k = j
					else:
						for l in range(len(predictions[i])):
							if(l < 10):
								predictions[i][l] = 0.0000000e+00
							else:
								predictions[i][l] = 1.0000000e+00
						k = 0
						break

		print("MODEL PREDICTIONS POST WORK -------------------------------------\n " + str(predictions))
		print("\n-----------------------------------------------------\n")

		indices2 = np.argmax(predictions,1)
		# print("\nLabels: " + str(labels) + "\n")
		# print("Indici: " + str(indices) + "\n")
		#print("--- " + str(lab) + " ---\n")
		lab = [labels[x] for x in indices2]
		self.show_imgs(X_test[:16],lab)

if __name__ == "__main__":
	 using = useNN()
	 using.fire()