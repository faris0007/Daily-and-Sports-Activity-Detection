import numpy as np
import matplotlib.pyplot as plt
from plotters import *
from extractData import get_data_for_solution_one
from evaluation import Evaluation
from extractData import get_data_for_solution_two




class KMeans:
    def __init__(self, k, n_times=5, max_iterations=100, ep=1e-6):
        self.k = k
        self.n_times = n_times
        self.max_iterations = max_iterations
        self.ep = ep

    def _euclidean_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def _k_means_algorithm(self, data):
        centroids_indices = np.random.choice(len(data), self.k, replace=False)
        centroids = data[centroids_indices]

        for _ in range(self.max_iterations):
            clusters = np.argmin(np.linalg.norm(data[:, None] - centroids, axis=2), axis=1)
            new_centroids = np.array([np.mean(data[clusters == i], axis=0) for i in range(self.k)])
            if np.all(np.abs(new_centroids - centroids) < self.ep):
                break
            centroids = new_centroids
        return clusters, centroids

    def fit_predict(self, data):
        best_clusters, best_centroids = None, None
        best_score = np.inf
        for _ in range(self.n_times):
            clusters, centroids = self._k_means_algorithm(data)
            within_cluster_score = 0
            for i in range(self.k):
                within_cluster_score += np.sum((data[clusters == i] - centroids[i]) ** 2)
            if within_cluster_score < best_score:
                best_score = within_cluster_score
                best_clusters, best_centroids = clusters, centroids

        return best_clusters, best_centroids


if __name__ == '__main__':
    data, _, data_labels, _ = get_data_for_solution_two()
    # data = np.array([
    #     [5, 8],
    #     [10, 8],
    #     [11, 8],
    #     [6, 7],
    #     [10, 7],
    #     [12, 7],
    #     [13, 7],
    #     [5, 6],
    #     [10, 6],
    #     [13, 6],
    #     [14, 6],
    #     [6, 5],
    #     [11, 5],
    #     [15, 5],
    #     [2, 4],
    #     [3, 4],
    #     [5, 4],
    #     [6, 4],
    #     [7, 4],
    #     [9, 4],
    #     [15, 4],
    #     [3, 3],
    #     [7, 3],
    #     [8, 2]
    # ])
    # data_labels=np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2 ])
    kmeans = KMeans(k=19)
    labels, _ = kmeans.fit_predict(data)
    plot_clusters(data, labels, 'Clusters')
    print(labels)
    evaluation = Evaluation(labels, data_labels)
    evaluation.evaluate()