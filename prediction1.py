import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications.xception import Xception
import cv2
import numpy as np
from keras.models import load_model


class Indian_Currency_To_Text:
    def __init__(self):
        pass

    def display_text(self, path_img, show_plot=False):
        self.path_img = path_img
        d = {0: "50", 1: "20", 2: "2000", 3: "10", 4: "100", 5: "500", 6: "200"}

        model2 = load_model("model_2.h5")
        model2.summary()
        model2.compile(
            optimizer=Nadam(0.00075),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=["accuracy"],
        )

        img = cv2.imread(self.path_img)
        plt.imshow(img)
        img = np.asarray(img) / (255.0)
        img = img.reshape(-1, 299, 299, 3)

        prediction = model2.predict(img)

        if show_plot == False:
            if np.array(prediction)[0, np.argmax(prediction)] >= 0.8:
                if np.argmax(prediction) != 2:
                    return d[np.argmax(prediction)] + " Rupees"
                elif np.argmax(prediction) == 2:
                    return d[np.argmax(prediction)] + " Rupees(No Longer In Use)"
            else:
                return "This Is Not Indian Currency"
            
            
        if show_plot == True:
            if np.array(prediction)[0, np.argmax(prediction)] >= 0.8:
                if np.argmax(prediction) != 2:
                    plt.title(d[np.argmax(prediction)] + " Rupees")
                    plt.show()
                    return d[np.argmax(prediction)] + " Rupees"
                elif np.argmax(prediction) == 2:
                    plt.title(d[np.argmax(prediction)] + " Rupees(No Longer In Use)")
                    plt.show()
                    return d[np.argmax(prediction)] + " Rupees(No Longer In Use)"
            else:
                plt.title("This Is Not Indian Currency")
                plt.show()
                return "This Is Not Indian Currency"

## Use Of the Class

Currency = Indian_Currency_To_Text()
print(Currency.display_text(path_img = '10_test.jpg',show_plot = True))