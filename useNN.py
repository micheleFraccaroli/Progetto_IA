import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot

import tensorflow as tf
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.preprocessing import image
from keras.utils import plot_model
from Load_Module import Load_Module
from scipy.misc import toimage
from tensorflow import keras

# hide AVX2 warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def animate():
    done = 0
    while done != 4:
        sys.stdout.write('\rRecognition in progress |')
        time.sleep(0.1)
        sys.stdout.write('\rRecognition in progress /')
        time.sleep(0.1)
        sys.stdout.write('\rRecognition in progress -')
        time.sleep(0.1)
        sys.stdout.write('\rRecognition in progress \\')
        time.sleep(0.1)
        done = done + 1


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    MAGENTA = '\033[1;35m'
    CYAN = '\033[1;36m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


class useNN:

    def show_imgs(self, X, lab):
        pyplot.figure(1)
        k = 0
        for i in range(0, 4):
            for j in range(0, 5):
                pyplot.subplot2grid((4, 5), (i, j))
                pyplot.imshow(toimage(X[k]))
                pyplot.annotate(lab[k], xy=(0, 0), xytext=(.8, -2),
                                fontsize=10, fontweight='bold', color='r')
                k = k+1
        # show the plot
        pyplot.subplots_adjust(hspace=1, wspace=1)
        pyplot.show()
    
    def fire(self):

        # LOAD DATA AND MODEL -------------------------------------------------
        lm = Load_Module()
        X_test = lm.loading()

        X_test = X_test.reshape(X_test.shape[0], 28, 28, 3)

        json_file = open('Models/6-Decider_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('Weights/6-Decider_weights.h5')

        # PLOTTING RESULT ----------------------------------------------------

        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

        indices = np.argmax(model.predict(X_test[:20]), 1)
        predictions = model.predict(X_test[:20])

        indices2 = np.argmax(predictions, 1)
        lab = [labels[x] for x in indices]
        self.show_imgs(X_test[:20], lab)

if __name__ == "__main__":

    os.system('clear')
    print(bcolors.MAGENTA + "       _______ _     _ __   _ _______ _     _       _____  _______ ______  " + bcolors.ENDC)
    print(bcolors.OKBLUE + "|      |_____| |     | | \  | |       |_____|      |_____] |_____| |     \ " + bcolors.ENDC)
    print(bcolors.CYAN + "|_____ |     | |_____| |  \_| |_____  |     |      |       |     | |_____/ \n\n" + bcolors.ENDC)

    animate()
    print("\n")

    using = useNN()
    using.fire()
