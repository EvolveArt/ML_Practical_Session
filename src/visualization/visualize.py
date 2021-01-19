import numpy as np
import matplotlib.pyplot as plt

X = np.load("../data/external/MNIST_X_28x28.npy")
Y = np.load("../data/external/MNIST_y.npy")


def plot_samples(rows=4, cols=5):
    """
      Plot a few image samples of the dataset
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, cols * rows + 1):
        img_index = np.random.randint(len(X))
        fig.add_subplot(rows, cols, i)
        plt.imshow(X[img_index])
        plt.title(f"Classe {str(Y[img_index])}")

    plt.show()
    plt.clf()


plot_samples()
