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
    labels = np.zeros(len(data))
    for i, cluster in enumerate(best_clusters):
        labels[[data.tolist().index(point.tolist()) for point in cluster]] = i
    return labels


if __name__ == '__main__':
    data = np.array([
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
    print(k_means(data, 3))
    plot_clusters(data, k_means(data, 3), 'Clusters')
