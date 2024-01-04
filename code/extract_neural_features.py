import numpy as np
import pvml
import matplotlib.pyplot as plt
import os

classes = os.listdir("images/test")
classes.sort()


def extract_neural_features(im, net):
    activations = net.forward(im[None, :, :, :])  # aggiungo una dimensione in più, passo da 224x244x3 a 1x224x244x3
    features = activations[-3]
    # features = np.average(features, axis=(0, 1))
    features = features.reshape(-1)
    return features


def process_directory(path, net):
    all_features = []
    all_labels = []
    for klass_label, klass in enumerate(classes):
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0
            features = extract_neural_features(image, net)
            all_features.append(features)
            all_labels.append(klass_label)
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


cnn = pvml.CNN.load("pvmlnet.npz")

X, Y = process_directory("images/test", cnn)
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)


X, Y = process_directory("images/train", cnn)
print("train", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)
