import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

dense_layer = 1
layer_size = 32
conv_layer = 2
pixel = 200
batch_sizes = 32


#  Model
for batch_size in range(1):
    NAME = "{}-batch-{}".format(batch_size, int(time.time()))
    tensorboard = TensorBoard(log_dir='logs/3-conv-32-nodes-1-dense/{}'.format(NAME))

    X = pickle.load(open("X_{}.pickle".format(int(pixel)), "rb"))
    y = pickle.load(open("y_{}.pickle".format(int(pixel)), "rb"))

    model = Sequential()
    model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for l in range(conv_layer - 1):
        model.add(Conv2D(layer_size, (3, 3)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Converts 3D feature maps into 1D feature vectors

    for l in range(dense_layer):
        model.add(Dense(layer_size))
        model.add(Activation('relu'))

    model.add(Dense(4))
    model.add(Activation('sigmoid'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=["accuracy"])

    #  Run Model
    model.fit(X,
              y,
              epochs=10,
              batch_size=batch_size,
              validation_split=0.1,
              callbacks=[tensorboard])
