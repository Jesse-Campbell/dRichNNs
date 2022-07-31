import os
from numpy import load
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from spektral.data import Dataset, DisjointLoader, Graph, SingleLoader
#from spektral.layers import GCSConv, GlobalAvgPool
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.models import GeneralGNN

from spektral.layers import *

######################################
# Config
######################################

learning_rate = 1e-1  # Learning rate
epochs = 400  # Number of training epochs
es_patience = 10  # Patience for early stopping
batch_size = 128  # Batch size

######################################
# Create Dataset
######################################


class GraphsDataset(Dataset):
    def __init__(self, adjacency_distance, randomize=False, **kwargs):
        self.adjacency_distance = adjacency_distance
        super().__init__()

    def read(self):
        electrons = os.path.join(os.path.expanduser('~/Desktop/rich-detector-analysis/data/graphData'), 'e- single')
        electronList = np.random.choice([f for f in os.listdir(electrons)], 500)

        kaons = os.path.join(os.path.expanduser('~/Desktop/rich-detector-analysis/data/graphData'), 'kaon+ single')
        kaonList = np.random.choice([f for f in os.listdir(kaons)], 500)

        pions = os.path.join(os.path.expanduser('~/Desktop/rich-detector-analysis/data/graphData'), 'pi+ single')
        pionList = np.random.choice([f for f in os.listdir(pions)], 500)

        protons = os.path.join(os.path.expanduser('~/Desktop/rich-detector-analysis/data/graphData'), 'proton single')
        protonList = np.random.choice([f for f in os.listdir(protons)], 500)

        data = [electronList, kaonList, pionList, protonList]
        folders = [electrons, kaons, pions, protons]
        output = []

        for i in range(len(data)):
            for event in data[i]:
                if event.endswith('.npz'):
                    graph = load(os.path.join(folders[i], event), allow_pickle=True)
                    x, y = graph['x'], graph['y']
                    a = self.create_adjacency_matrix(x)
                    #a = self.calculate_distances(a)
                    a = sp.csr_matrix(a)
                    n = len(x)
                    new_x = np.zeros((n, 3), dtype=float)
                    for k in range(n):
                        new_x[k] = (x[k][2], x[k][3], x[k][4])
                    output.append(Graph(x=new_x, a=a, y=y))

        np.random.shuffle(output)
        return output

    def create_adjacency_matrix(self, x):
        n = len(x)
        a = np.zeros((n, n), dtype=float)
        for i in range(n):
            for j in range(n):
                a[i][j] = np.sqrt(np.square(x[i][0] - x[j][0]) + np.square(x[i][1] - x[j][1]))
        return a

    def calculate_distances(self, a):
        for i in range(len(a)):
            for j in range(len(a[i])):
                if a[i][j] < self.adjacency_distance:
                    a[i][j] = 1
                else:
                    a[i][j] = 0
        return a


######################################
# Load Data
######################################

data = GraphsDataset(1, transforms=NormalizeAdj())

# Train/validation/test split
indexes = np.random.permutation(len(data))
split_va, split_te = int(0.8 * len(data)), int(0.9 * len(data))
idx_tr, idx_va, idx_te = np.split(indexes, [split_va, split_te])
data_tr = data[idx_tr]
data_va = data[idx_va]
data_te = data[idx_te]

# Data loaders
loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
loader_va = DisjointLoader(data_va, batch_size=batch_size)
loader_te = DisjointLoader(data_te, batch_size=batch_size)

######################################
# Build model
######################################


class Net(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCSConv(32, activation="sigmoid")
        #self.conv2 = GCSConv(64, activation="sigmoid")
        #self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(data.n_labels, activation="softmax")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        #x = self.conv2([x, a])
        #x = self.conv3([x, a])
        output = self.global_pool([x, i])
        output = self.dense(output)

        return output


model = Net()

#model = GeneralGNN(data.n_labels, activation="softmax")

optimizer = Adam(lr=learning_rate)
loss_fn = CategoricalCrossentropy()

################################################################################
# Fit model
################################################################################


@tf.function(input_signature=loader_tr.tf_signature(), experimental_relax_shapes=True)
def train_step(inputs, target):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)
        loss = loss_fn(target, predictions) + sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    acc = tf.reduce_mean(categorical_accuracy(target, predictions))
    return loss, acc


def evaluate(loader):
    output = []
    step = 0
    while step < loader.steps_per_epoch:
        step += 1
        inputs, target = loader.__next__()
        pred = model(inputs, training=False)
        outs = (
            loss_fn(target, pred),
            tf.reduce_mean(categorical_accuracy(target, pred)),
            len(target),  # Keep track of batch size
        )
        output.append(outs)
        if step == loader.steps_per_epoch:
            output = np.array(output)
            return np.average(output[:, :-1], 0, weights=output[:, -1])


epoch = step = 0
best_val_loss = np.inf
best_weights = None
patience = es_patience
results = []
for batch in loader_tr:
    step += 1
    loss, acc = train_step(*batch)
    results.append((loss, acc))
    if step == loader_tr.steps_per_epoch:
        step = 0
        epoch += 1

        # Compute validation loss and accuracy
        val_loss, val_acc = evaluate(loader_va)
        print(
            "Ep. {} - Loss: {:.3f} - Acc: {:.3f} - Val loss: {:.3f} - Val acc: {:.3f}".format(
                epoch, *np.mean(results, 0), val_loss, val_acc
            )
        )

        # Check if loss improved for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = es_patience
            print("New best val_loss {:.3f}".format(val_loss))
            best_weights = model.get_weights()
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping (best val_loss: {})".format(best_val_loss))
                break
        results = []

################################################################################
# Evaluate model
################################################################################

model.set_weights(best_weights)  # Load best model
test_loss, test_acc = evaluate(loader_te)
print("Done. Test loss: {:.4f}. Test acc: {:.2f}".format(test_loss, test_acc))
