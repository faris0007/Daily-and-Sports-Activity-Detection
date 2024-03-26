from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from plotters import *

from extractData import get_data_for_solution_one
from evaluation import Evaluation
class NormalizedCut:

    def __init__(self, k=3, gamma=1):
        self.k = k
        self.gamma = gamma
        self.threshold = 1e-6
    def _sort_eigenvalues_and_eigenvectors(self, eigenvalues, eigenvectors):
        idx = eigenvalues.argsort()[::]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        return eigenvalues, eigenvectors

    def fit_predict(self, X):
        A = rbf_kernel(X, gamma=self.gamma)
        A = np.where(A < self.threshold, 0, A)
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        La = np.dot(np.linalg.inv(D), L)
        eigenvalues, eigenvectors = np.linalg.eig(La)
        eigenvalues, eigenvectors = self._sort_eigenvalues_and_eigenvectors(eigenvalues, eigenvectors)
        U = eigenvectors[:, :self.k]
        N = np.linalg.norm(U, axis=1, keepdims=True)
        N[N == 0] = 1e-6  # Replace zeros in N with a very small value
        U_real = np.real(U)  # Extract real part of U
        Y = U_real / N
        labels = KMeans(n_clusters=self.k).fit_predict(Y)
        return labels


if __name__ == "__main__":
    data_matrix, _, data_labels, _ = get_data_for_solution_one()
    normalized_cut = NormalizedCut(19, .01)
    labels = normalized_cut.fit_predict(data_matrix)
    # plot_clusters(data_matrix, labels, "Clustering")
    print(labels)
    evaluation = Evaluation(labels, data_labels)
    evaluation.evaluate()
