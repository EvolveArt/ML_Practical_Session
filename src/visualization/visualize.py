import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import mode

from src.data.make_dataset import X_raw, Y


def plot_variance_explained(variance_explained):
    """
    Plots variance explained for each component

    args:
		variance_explained : 1d array of the explained_variance for each component
    """

    plt.figure()
    plt.plot(np.arange(1, len(variance_explained) + 1), variance_explained, "--k")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()


def plot_samples(X=X_raw, rows=4, cols=5):
    """
      Plot a few image samples of the dataset
    """
    fig = plt.figure(figsize=(8, 8))

    for i in range(1, cols * rows + 1):
        img_index = np.random.randint(len(X))
        ax = fig.add_subplot(rows, cols, i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(X[img_index], cmap="gray")
        plt.title(f"Classe {str(Y[img_index])}")

    plt.show()
    plt.clf()


def plot_MNIST_reconstruction(X_old, X_new):
    """
    Plots 9 images of the MNIST dataset side-by-side with the modified images.
    """
    plt.figure()

    # Raw Images axis
    ax = plt.subplot(121)
    k = 0
    for k1 in range(3):
        for k2 in range(3):
            k = k + 1
            plt.imshow(
                X_old[k].reshape(28, 28),
                extent=[(k1 + 1) * 28, k1 * 28, (k2 + 1) * 28, k2 * 28],
                vmin=0,
                vmax=1,
                cmap="gray",
            )

    plt.xlim((3 * 28, 0))
    plt.ylim((3 * 28, 0))
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, labelbottom=False
    )
    ax.set_xticks([])
    ax.set_yticks([])
    plt.title("Raw Images")

    # Reconstructed Images axis
    ax = plt.subplot(122)
    k = 0
    for k1 in range(3):
        for k2 in range(3):
            k = k + 1
            plt.imshow(
                X_new[k].reshape(28, 28),
                extent=[(k1 + 1) * 28, k1 * 28, (k2 + 1) * 28, k2 * 28],
                vmin=0,
                vmax=1,
                cmap="gray",
            )

    plt.xlim((3 * 28, 0))
    plt.ylim((3 * 28, 0))
    plt.tick_params(
        axis="both", which="both", bottom=False, top=False, labelbottom=False
    )
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title("Reconstructed Images")

    plt.tight_layout()


def getClusterStats(cluster):
    class_freq = np.zeros(10)
    for i in range(10):
        class_freq[i] = np.count_nonzero(cluster == i)
    most_freq = np.argmax(class_freq)
    n_majority = np.max(class_freq)
    n_all = np.sum(class_freq)
    n_confidence = float(n_majority / n_all)
    return (most_freq, n_confidence)


def getClustersStats(y_pred, y_true):
    stats = np.zeros((10, 2))
    for i in range(10):
        indices = np.where(y_pred == i)
        cluster = y_true[indices]
        stats[i, :] = getClusterStats(cluster)
    return stats


def plot_digits_rows(digits, title, labels):
    n = digits.shape[0]
    n_rows = n / 25 + 1
    n_cols = 25
    plt.figure(figsize=(n_cols * 0.9, n_rows * 1.3))
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(title)
    for i in range(n):
        plt.subplot(n_rows, n_cols, i + 1)
        plot_digit(digits[i, :], "%d" % labels[i])


def plot_digit(digit, label):
    plt.axis("off")
    plt.imshow(digit.reshape((28, 28)), cmap="gray")
    plt.title(label)


def plotClusters(X, y_pred, y_true, stats):
    for i in range(10):
        indices = np.where(y_pred == i)
        title = f"Most frequent digit : {stats[i, 0]} / Cluster confidence : {stats[i, 1]:.2f}"
        plot_digits_rows(X[indices][:25], title, y_true[indices])


def plotConfusionMatrix(y_pred, y_true):

    labels = np.zeros_like(y_pred)

    for i in range(10):
        mask = y_pred == i
        labels[mask] = mode(y_true[mask])[0]

    mat = confusion_matrix(y_true, labels)
    sns.heatmap(mat.T, square=True, annot=True, fmt="d", cbar=False)

    plt.xlabel("True digit")
    plt.ylabel("Predicted digit")
    plt.show()


def plotClustersDigits(cluster_centers):
    fig, ax = plt.subplots(2, 5, figsize=(8, 3))
    centers = cluster_centers.reshape(10, 28, 28)

    for axi, center in zip(ax.flat, centers):
        axi.set(xticks=[], yticks=[])
        axi.imshow(center, interpolation="nearest", cmap=plt.cm.gray)

