import numpy as np
import pvml
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os

def load_data():
    data = np.loadtxt("test.txt.gz")
    Xtest = data[:, :-1]
    Ytest = data[:, -1].astype(int)
    return Xtest, Ytest

def print_probability_range(probs, start, end):
    for i in range(start, end):
        print(probs[i], "\t", i, "\n")

def compute_confusion_matrix(Xtest, Ytest, labels):
    cm = np.zeros((15, 15))
    for i in range(Xtest.shape[0]):
        cm[Ytest[i], labels[i]] += 1
    cm = cm / cm.sum(1, keepdims=True)
    return cm

def print_confusion_matrix(classes, cm):
    print(" " * 10, end="")
    for i in range(15):
        print(classes[i][:3], end="\t")
    print()

    for i in range(15):
        print("%9s" % classes[i], end=" ")
        for j in range(15):
            print("%.3f" % cm[i, j], end=" ")
        print()

def plot_heatmap(classes, cm):
    for i in range(15):
        for j in range(15):
            cm[i, j] = int(100 * cm[i, j])
    df_cm = pd.DataFrame(cm, classes, classes)
    sn.set(font_scale=1)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 10}, xticklabels=classes, yticklabels=classes)
    plt.show()

def main():
    classes = os.listdir("images/test")
    classes.sort()

    Xtest, Ytest = load_data()

    mlp = pvml.MLP.load("cakes-mlp.npz")
    labels, probs = mlp.inference(Xtest)

    print("\n\n\n")
    print_probability_range(probs, 80, 100)
    print("\n\n\n")

    cm = compute_confusion_matrix(Xtest, Ytest, labels)
    print_confusion_matrix(classes, cm)

    plot_heatmap(classes, cm)

if __name__ == '__main__':
    main()
