''''
    Author: 
        Ahmed Rafik El-Mehdi BAAHMED || BR34CH-HUNT3R

'''

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
import tensorflow as tf
import os

'''
# -------------------------- To use WandB uncomment this section ---------------------------------
import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config
'''

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data to improve the training
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

# shape of each image (28x28)
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1]

# create model
model=Sequential()
model.add(Flatten(input_shape=(img_width,img_height)))

#   This means that we will train num_classes (= 10) perceptrons, so our output will be 10 outputs (each output for an integer)
#   softmax function as an activation function
model.add(Dense(num_classes, activation="softmax"))

# categorical_crossentropy loss function to improve our training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
# to use WandB uncomment the next line and comment the one after.
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), callbacks=[WandbCallback(labels=labels, data_type="image")])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# save the model
# serialize model to json
model_json = model.to_json()
with open("handwritten_digits_recognition_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("handwritten_digits_recognition_model.h5")

'''
    Useful ideas to improve accuracy : 
        1. Set activation function -> softmax
        2. Set loss function -> categorical cross entropy
        3. Normalized data
'''