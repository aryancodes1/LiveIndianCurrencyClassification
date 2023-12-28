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


d = {0:'50',1:'20',2:'2000',3:'10',4:'100',5:'500',6:'200'}

model2 = load_model('model_2.h5')
model2.summary()
model2.compile(
    optimizer=Nadam(0.00075),
    loss=tf.keras.losses.sparse_categorical_crossentropy,
    metrics=["accuracy"]
)


img = cv2.imread('/Users/arunkaul/Desktop/Currency ML Project/10_test.jpg')
plt.imshow(img)

img = np.asarray(img) / (255.0)
img = img.reshape(-1, 299, 299, 3)

prediction = model2.predict(img)
plt.title(d[np.argmax(prediction)]+" Rupees")
plt.show()