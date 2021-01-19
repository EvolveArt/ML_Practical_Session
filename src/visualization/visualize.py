import numpy as np
import matplotlib.pyplot as plt

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

