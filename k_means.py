import numpy as np


def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))


def k_means_algorithm(data, k, max_iterations=100, ep=0.00001):
    centroids_indices = np.random.choice(len(data), k, replace=False)

    centroids = data[centroids_indices]

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            nearest_centroid_index = np.argmin(distances)
            clusters[nearest_centroid_index].append(point)
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        if np.sum(np.abs(centroids - new_centroids)) < ep:
            break
        centroids = new_centroids

    return clusters, centroids


def k_means(data, k, n_times=5, max_iterations=100, ep=0.00001):
    best_clusters, best_centroids = None, None
    best_score = np.inf
    for _ in range(n_times):
        clusters, centroids = k_means_algorithm(data, k, max_iterations, ep)
        within_cluster_score = 0
        for cluster, centroid in zip(clusters, centroids):
            within_cluster_score += np.sum([euclidean_distance(point, centroid) for point in cluster])
        if within_cluster_score < best_score:
            best_score = within_cluster_score
            best_clusters, best_centroids = clusters, centroids
    return best_clusters, best_centroids
