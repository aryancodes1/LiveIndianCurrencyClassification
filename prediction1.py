import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
import tensorflow as tf


model = load_model("model_2.h5")
d = {0: "50", 1: "20", 2: "2000", 3: "10", 4: "100", 5: "500", 6: "200"}
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"],
)
frameWidth = 680
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

while cap.isOpened():
    success, img = cap.read()
    if success:
        img = cv2.flip(img, 1)
        img = cv2.resize(img, (299, 299))
        cv2.imshow("Result", img)
        img = np.array(img)
        img = img.reshape(1, 299, 299, 3)
        img = img / 255.0
        if 0xFF == ord("q"):
            break
        elif cv2.waitKey(1):
            pred = model.predict(img, verbose=0)[0]
            if pred[np.argmax(pred)] >= 0.8:
                cv2.setWindowTitle("Result", title=d[np.argmax(pred)])
                print(pred[np.argmax(pred)])
            else:
                cv2.setWindowTitle("Result", title="No Currency")
