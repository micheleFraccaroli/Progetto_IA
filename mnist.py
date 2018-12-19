from time import time
import tensorflow as tf
import numpy as np
import pickle
from scipy.misc import toimage
from tensorflow import keras
#from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential, load_model, Model, model_from_json
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.python.keras.utils import np_utils, to_categorical
from tensorflow.python.keras.optimizers import RMSprop, SGD
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras import backend as K

#K.tensorflow_backend._get_available_gpus()
if K.backend()=='tensorflow':
    K.set_image_data_format("channels_last")

print("GPU: " + str(tf.test.is_gpu_available()))

num_classes = 10+1
batch_size = 96
epochs = 50

#(X,Y),(X_test,Y_test) = mnist.load_data()
data = np.load('mnist.npz')
X = data['x_train']
Y = data['y_train']
X_test = data['x_test']
Y_test = data['y_test']

#if K.image_data_format() == 'channels_first':
print("X shape[0] --> " + str(X.shape[0]))
print("X shape --> " + str(X.shape))
X = X.reshape(X.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0],28, 28,1)
input_shape = (28, 28,1)

# Convert class vectors to binary class matrices.
Y = np_utils.to_categorical(Y, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
X = X.astype('float32')
X_test = X_test.astype('float32')
X  /= 255
X_test /= 255

# Architecture of neural network
# -- --
def model_cnn():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(Conv2D(64,(3,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(128))
	model.add(Activation('relu'))

	model.add(Dropout(0.5))
	model.add(Dense(num_classes))
	model.add(Activation('softmax'))
	# -- end network --

	# Compiling the model
	#opt = RMSprop(lr=0.0001, decay=1e-6)
	lrate = 0.01
	decay = lrate/epochs
	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	model.compile(optimizer=sgd,
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	return model


# Training
def training():
	cnn_i = model_cnn()
	tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
	cnn = cnn_i.fit(X,Y,batch_size=batch_size,epochs=epochs,validation_data=(X_test,Y_test),shuffle=True,callbacks=[tensorboard])

	# print("\n----- Real-time data augmentation -----\n")
	# datagen = ImageDataGenerator(
	# 	rotation_range=40,
 #        rescale=1./255,
 #        shear_range=0.2,
 #        zoom_range=0.2,
 #        horizontal_flip=True,
 #        fill_mode='nearest')

	# datagen.fit(X)
	# cnn_i.fit_generator(datagen.flow(X,Y,batch_size=batch_size),epochs=epochs,steps_per_epoch=len(X)//batch_size, validation_data=(X_test,Y_test),workers=2)

	scores = cnn_i.evaluate(X_test, Y_test)
	print('Loss: %.3f' % scores[0])
	print('Accuracy: %.3f' % (scores[1]*100))

	# Saving model
	# Save to disk
	model_json = cnn_i.to_json()
	with open('mnist_model.json', 'w') as json_file:
	    json_file.write(model_json)

	cnn_i.save_weights('mnist.h5')


if __name__ == '__main__':
	training()
