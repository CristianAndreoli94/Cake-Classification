import numpy as np
import pvml
import matplotlib.pyplot as plt

data = np.loadtxt("train.txt.gz")
X = data[:, :-1]
Y = data[:, -1].astype(int)

data = np.loadtxt("test.txt.gz")
Xtest = data[:, :-1]
Ytest = data[:, -1].astype(int)

nclasses = Y.max() + 1

mlp = pvml.MLP([X.shape[1], nclasses])
epochs = 5000
batch_size = 50
lr = 0.0001

train_accs = []
test_accs = []
plt.ion()
for epoch in range(epochs):
    steps = X.shape[0] // batch_size
    mlp.train(X, Y, lr=lr, batch=batch_size, steps=steps)
    plt.figure(1)
    if epoch % 100 == 0:
        predictions, probs = mlp.inference(X)
        train_acc = (predictions == Y).mean()
        train_accs.append(train_acc * 100)
        predictions, probs = mlp.inference(Xtest)
        test_acc = (predictions == Ytest).mean()
        test_accs.append(test_acc * 100)
        print(epoch, train_acc * 100, test_acc * 100)
        plt.clf()
        plt.plot(train_accs)
        plt.plot(test_accs)
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")
        plt.legend(["train", "test"])
        plt.pause(0.05)

mlp.save("cakes-mlp.npz")
    
plt.ioff()
plt.show()
