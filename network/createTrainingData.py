import numpy as np
import os
import cv2
import random
import pickle
import matplotlib.pyplot as plt

DATADIR = os.path.expanduser("~/Desktop/rich-detector-analysis/data/eventData/mapData")
CATEGORIES = ["e- single", "kaon+ single", "pi+ single", "proton single"]
ARRAY_SIZE = 100

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        label = np.zeros(4, dtype=int)
        class_num = CATEGORIES.index(category)
        label[class_num] = 1
        for file in np.random.choice([f for f in os.listdir(path)], 500):
            with open(os.path.join(path, file), newline="") as csv_file:
                try:
                    x = []
                    y = []
                    array = np.genfromtxt(csv_file, delimiter=",")
                    new_array = cv2.resize(array, (ARRAY_SIZE, ARRAY_SIZE))
                    training_data.append([new_array, label])

                except Exception as e:
                    pass

    random.shuffle(training_data)


create_training_data()


X = []
y = []

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, ARRAY_SIZE, ARRAY_SIZE, 1)
y = np.array(y)

pickle_out = open("X_{}.pickle".format(ARRAY_SIZE), "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y_{}.pickle".format(ARRAY_SIZE), "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#pickle_in = open("X_500.pickle", "rb")
#X = pickle.load(pickle_in)