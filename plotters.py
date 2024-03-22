import numpy as np
import matplotlib.pyplot as plt


def plot_clusters(X, labels, title):
    plt.figure(figsize=(8, 6))

    for label in np.unique(labels):
        indices = labels == label
        plt.scatter(X[indices, 0], X[indices, 1], label=f'Cluster {label + 1}')

    plt.title(title)
    plt.legend()
    plt.show()
