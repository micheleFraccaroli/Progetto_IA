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


def show_imgs(X,lab):
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

def usage():
	data = np.load('mnist.npz')
	X = data['x_train']
	Y = data['y_train']
	X_test = data['x_test']
	Y_test = data['y_test']

	print("X_TEST SHAPE ----> " + str(X_test.shape))

	#if K.image_data_format() == 'channels_first':
	X = X.reshape(X.shape[0], 28, 28,1)
	X_test = X_test.reshape(X_test.shape[0],28, 28,1)
	input_shape = (28, 28,1)

	# # Convert class vectors to binary class matrices.
	# Y = np_utils.to_categorical(Y, num_classes)
	# Y_test = np_utils.to_categorical(Y_test, num_classes)
	# X = X.astype('float32')
	# X_test = X_test.astype('float32')
	# X  /= 255
	# X_test /= 255

	#json_file = open('json_model/model_cluster_cnn2mod.json', 'r')
	json_file = open('mnist_model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	model = model_from_json(loaded_model_json)
	#model.load_weights('h5/bird_keras_cnn2mod.h5')
	model.load_weights('mnist.h5')

	#labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
	labels = ['0','1','2','3','4','5','6','7','8','9','N']
	 
	indices = np.argmax(model.predict(X_test[:16]),1)
	res = model.predict(X_test[:16])
	#print("Predizione → " + str(res))
	print("\nLabels: " + str(labels) + " ---\n")
	print("Indici: " + str(indices) + " ---\n")
	lab = [labels[x] for x in indices]
	print("--- " + str(lab) + " ---\n")
	show_imgs(X_test[:16],lab)

usage()