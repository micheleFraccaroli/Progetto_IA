import os
from time import time
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras.optimizers import Adam
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


class Decider:
	if K.backend()=='tensorflow':
	    K.set_image_data_format("channels_last")

	print("GPU: " + str(tf.test.is_gpu_available()))

	def launching(self,epochs):
		num_classes = 10
		batch_size = 96
		epochs = 800

		# Convolutional Neural Network -----

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
		model.add(Conv2D(64,(3,3)))
		model.add(Activation('relu'))
		#model.add(Dropout(0.5))
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

		# END CNN -----
		adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
		model.compile(loss='categorical_crossentropy',
	              optimizer=adam,
	              metrics=['accuracy'])

		# augmentation configuration for training
		train_datagen = ImageDataGenerator(
		        rescale=1./255,
		        shear_range=0.2,
		        zoom_range=0.2,
		        #brightness_range = [0.03,0.06],
		        height_shift_range = 0.1,
		        vertical_flip = True)

		# augmentation configuration for testing:
		test_datagen = ImageDataGenerator(rescale=1./255,
				#brightness_range = [0.03,0.06],
		        height_shift_range = 0.1,
		        vertical_flip = True)

		# read data from local folder and generate augmented data for training session
		train_generator = train_datagen.flow_from_directory(
		        'DigitDataset/Images/Train/',
		        target_size=(28, 28),
		        batch_size=batch_size,
		        class_mode='categorical')

		# read data from local folder and generate augmented data for testing session
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
		with open('Decider_model.json', 'w') as json_file:
		    json_file.write(model_json)

		model.save_weights('decider_weights.h5')

if __name__ == '__main__':

	os.system('clear')
	print(bcolors.MAGENTA + "\n\n________                .__    .___            " + bcolors.ENDC)
	print(bcolors.MAGENTA + "\______ \   ____   ____ |__| __| _/___________ " + bcolors.ENDC)
	print(bcolors.OKBLUE + " |    |  \_/ __ \_/ ___\|  |/ __ |/ __ \_  __ \ " + bcolors.ENDC)
	print(bcolors.OKBLUE + " |    `   \  ___/\  \___|  / /_/ \  ___/|  | \/" + bcolors.ENDC)
	print(bcolors.CYAN + "/_______  /\___  >\___  >__\____ |\___  >__|   " + bcolors.ENDC)
	print(bcolors.CYAN + "        \/     \/     \/        \/    \/       \n\n" + bcolors.ENDC)

	c = Decider()
	c.launching(epochs)
