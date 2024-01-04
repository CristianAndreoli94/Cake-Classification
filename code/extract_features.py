import numpy as np
import matplotlib.pyplot as plt
import os
import image_features

classes = os.listdir("images/test")
classes.sort()


def process_directory(path):
    all_features = []
    all_labels = []
    klass_label = 0
    for klass in classes:
        image_files = os.listdir(path + "/" + klass)
        for imagename in image_files:
            image_path = path + "/" + klass + "/" + imagename
            image = plt.imread(image_path) / 255.0

            features1 = image_features.color_histogram(image)
            features2 = image_features.edge_direction_histogram(image)
            features3 = image_features.cooccurrence_matrix(image)

            features1 = features1.reshape(-1)
            features2 = features2.reshape(-1)
            features3 = features2.reshape(-1)
            features = np.concatenate([features1, features2, features3])


            all_features.append(features)
            all_labels.append(klass_label)
        klass_label += 1
    X = np.stack(all_features, 0)
    Y = np.array(all_labels)
    return X, Y


X, Y = process_directory("images/test")
print("test", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("test.txt.gz", data)


X, Y = process_directory("images/train")
print("train", X.shape, Y.shape)
data = np.concatenate([X, Y[:, None]], 1)
np.savetxt("train.txt.gz", data)
