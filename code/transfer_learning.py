import numpy as np
import pvml


cnn = pvml.CNN.load("pvmlnet.npz")
mlp = pvml.MLP.load("cakes-mlp.npz")


cnn.weights[-1] = mlp.weights[0][None, None, :, :]
cnn.biases[-1] = mlp.biases[0]

cnn.save("cakes-cnn.npz")
