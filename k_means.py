import numpy as np

class KMeans:
    def __init__(self, k, n_times=5, max_iterations=100, ep=1e-6):
        self.k = k
        self.n_times = n_times
        self.max_iterations = max_iterations
        self.ep = ep

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
