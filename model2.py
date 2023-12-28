from img_load_class import ImageLoad
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from tensorflow.keras.losses import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications.xception import Xception


d = {0:'50',1:'20',2:'2000',3:'10',4:'100',5:'500',6:'200'}

path_tr = "/Users/arunkaul/Desktop/Training Data"
path_ts = "/Users/arunkaul/Desktop/Testing Data"

train = ImageLoad(path_tr, 299,"Tr")
test = ImageLoad(path_ts, 299,"Ts")

(train_images, train_labels) = train.load_dataset()
(test_images, test_labels) = test.load_dataset()


#Model

model = Sequential()
model.add(Xception(include_top=False, weights="imagenet", pooling="max"))
model.add(Dense(1000, activation="relu"))
model.add(Dense(7, activation="softmax"))
model.layers[0].trainable = False

model.summary()

model.compile(
    optimizer=Nadam(0.001),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)
model.fit(train_images, train_labels, epochs=15,validation_data = (test_images,test_labels))
model.save("model_2.h5")


