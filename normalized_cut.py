from sklearn.metrics import pairwise_distances
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from plotters import *
from sklearn.cluster import SpectralClustering

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
        Y = U_real /N[:, np.newaxis]
        labels = KMeans(n_clusters=self.k).fit_predict(Y)
        return labels


from sklearn.metrics.cluster import contingency_matrix


def purity_score(y_true, y_pred):
    contingency = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)


if __name__ == "__main__":
    data_matrix, _, data_labels, _ = get_data_for_solution_one()
    normalized_cut = NormalizedCut(19, .000001)
    labels = normalized_cut.fit_predict(data_matrix)
    # plot_clusters(data_matrix, labels, "Clustering")
    print(labels)
    evaluation = Evaluation(labels, data_labels)
    evaluation.evaluate()
    # gammas = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    # for g in gammas:
    #     print("gamma", g)
    #     spectral = SpectralClustering(n_clusters=19, affinity='rbf', gamma=g, random_state=42)
    #
    #     # Fit and predict
    #     labels = spectral.fit_predict(data_matrix)
    #
    #     evaluation = Evaluation(labels, data_labels)
    #
    #     evaluation.evaluate()
    #     print("Purity Score")
    #     print(purity_score(data_labels, labels))
