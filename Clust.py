from time import time
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.python.keras import backend as K

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    MAGENTA = '\033[1;35m'
    CYAN = '\033[1;36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class Clust:
	if K.backend()=='tensorflow':
	    K.set_image_data_format("channels_last")

	print("GPU: " + str(tf.test.is_gpu_available()))

	def launching(self):
		num_classes = 10
		batch_size = 96
		epochs = 300

		# NET -----

		model = Sequential()
		model.add(Conv2D(32, (3, 3), input_shape=(28,28,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Conv2D(64,(3,3)))
		model.add(Activation('relu'))
		model.add(Conv2D(64,(3,3)))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		model.add(Conv2D(64,(3,3)))
		model.add(Activation('relu'))
		model.add(Conv2D(64,(3,3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		model.add(Flatten())
		model.add(Dropout(0.5))
		model.add(Dense(512))
		model.add(Dropout(0.5))
		model.add(Dense(512))
		model.add(Dropout(0.5))
		model.add(Dense(128))
		model.add(Activation('relu'))

		model.add(Dropout(0.5))
		model.add(Dense(num_classes))
		model.add(Activation('softmax'))

		# END -----

		model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

		# augmentation configuration for training
		train_datagen = ImageDataGenerator(
		        rescale=1./255,
		        shear_range=0.2,
		        zoom_range=0.2,
		        horizontal_flip=True)

		# augmentation configuration for testing:
		test_datagen = ImageDataGenerator(rescale=1./255)

		# this is a generator that will read pictures found in
		# subfolers of 'data/train', and indefinitely generate
		# batches of augmented image data
		train_generator = train_datagen.flow_from_directory(
		        'DigitDataset/Images/Train/',
		        target_size=(28, 28),
		        batch_size=batch_size,
		        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

		# this is a similar generator, for validation data
		validation_generator = test_datagen.flow_from_directory(
		        'DigitDataset/Images/Test/',
		        target_size=(28, 28),
		        batch_size=batch_size,
		        class_mode='categorical')

		#scores = model.evaluate(X_test, Y_test)
		#print('Loss: %.3f' % scores[0])
		#print('Accuracy: %.3f' % (scores[1]*100))

		model.fit_generator(
	        train_generator,
	        steps_per_epoch=2000 // batch_size,
	        epochs=epochs,
	        validation_data=validation_generator,
	        validation_steps=800 // batch_size)

		# Saving model
		model_json = model.to_json()
		with open('clust_model.json', 'w') as json_file:
		    json_file.write(model_json)

		model.save_weights('clust_weights.h5')

if __name__ == '__main__':

	print(bcolors.MAGENTA + "_________ .__                  __    " + bcolors.ENDC)
	print(bcolors.MAGENTA + "\_   ___ \|  |  __ __  _______/  |_  " + bcolors.ENDC)
	print(bcolors.OKBLUE + "/    \  \/|  | |  |  \/  ___/\   __\ " + bcolors.ENDC)
	print(bcolors.OKBLUE + "\     \___|  |_|  |  /\___ \  |  |   " + bcolors.ENDC)
	print(bcolors.CYAN + " \______  /____/____//____  > |__|   " + bcolors.ENDC)
	print(bcolors.CYAN + "        \/                \/         " + bcolors.ENDC)

	c = Clust()
	c.launching()
