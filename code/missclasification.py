import numpy as np
import pvml
import matplotlib.pyplot as plt
import os

cnn = pvml.CNN.load("pvmlnet.npz")
mlp = pvml.MLP.load("cakes-mlp.npz")

cnn.weights[-1] = mlp.weights[0][None, None, :, :]
cnn.biases[-1] = mlp.biases[0]

cnn.save("cakes-cnn.npz")

#imagepath = "cake-classification/images/test/ice_cream/45200.jpg"

classes = os.listdir("images/test")
classes.sort()
path = "images/test"

for klass in classes:
    counter = 0
    image_files = os.listdir(path + "/" + klass)
    for imagename in image_files:
        imagepath = path + "/" + klass + "/" + imagename

        image = plt.imread(imagepath) / 255.0
        labels, probs = cnn.inference(image[None, :, :, :])

        classes = os.listdir("images/test")
        classes = [c for c in classes if not c.startswith(".")]
        classes.sort()

        indices = (-probs[0]).argsort()
        for k in range(5):
            index = indices[k]
            # print(klass + " " + classes[index])
            # print(f"{k + 1} {classes[index]:10}  {probs[0][index] * 100:.1f}")
            if klass != classes[index] and k == 0:
                # print(f"{k + 1} {classes[index]:10}  {probs[0][index] * 100:.1f}")
                print(imagepath + " " + classes[index])
                counter += 1


    print(klass + " " + str(counter))

