from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel, linear_kernel
from plotters import *


class NormalizedCut:

    def __init__(self, k=3, gamma=1, coef0=1, degree=3, kernel='rbf'):
        self.k = k
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.threshold = 1e-6
        self.kernel = kernel

    def _compute_kernel_matrix(self, X):
        match self.kernel:
            case 'rbf':
                return rbf_kernel(X, gamma=self.gamma)
            case 'polynomial':
                return polynomial_kernel(X, degree=self.degree, coef0=self.coef0)
            case 'sigmoid':
                return sigmoid_kernel(X, gamma=self.gamma, coef0=self.coef0)
            case 'linear':
                return linear_kernel(X)
            case _:
                raise ValueError("Unsupported kernel type. Supported types: 'rbf', 'polynomial', 'sigmoid', 'linear'.")

    def _compute_eigenvalues_and_eigenvectors(self, matrix):
        return np.linalg.eig(matrix)

    def _sort_eigenvalues_and_eigenvectors(self, eigenvalues, eigenvectors):
        idx = eigenvalues.argsort()[::]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def fit_predict(self, X):
        A = self._compute_kernel_matrix(X)
        A = np.where(A < self.threshold, 0, A)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        La = np.dot(np.linalg.inv(D), L)
        eigenvalues, eigenvectors = np.linalg.eig(La)
        eigenvalues, eigenvectors = self._sort_eigenvalues_and_eigenvectors(eigenvalues, eigenvectors)
        U = eigenvectors[:, :self.k]
        Y = U / np.linalg.norm(U, axis=1, keepdims=True)
        labels = KMeans(n_clusters=self.k).fit_predict(Y)
        return labels


if __name__ == "__main__":
    data_matrix = np.array([
        [5, 8],
        [10, 8],
        [11, 8],
        [6, 7],
        [10, 7],
        [12, 7],
        [13, 7],
        [5, 6],
        [10, 6],
        [13, 6],
        [14, 6],
        [6, 5],
        [11, 5],
        [15, 5],
        [2, 4],
        [3, 4],
        [5, 4],
        [6, 4],
        [7, 4],
        [9, 4],
        [15, 4],
        [3, 3],
        [7, 3],
        [8, 2]
    ])

    normalized_cut = NormalizedCut(kernel='linear')
    labels = normalized_cut.fit_predict(data_matrix)
    plot_clusters(data_matrix, labels, "Clustering")
    print(labels)
