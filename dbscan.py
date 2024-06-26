import numpy as np
import matplotlib.pyplot as plt
from plotters import *
from sklearn.cluster import DBSCAN as DBSCANB


class DBSCAN:
    def __init__(self, epsilon, minpts):
        self.epsilon = epsilon
        self.minpts = minpts

    def _euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def get_neighbors(self, X, query_point):
        neighbors = []
        for i, point in enumerate(X):
            if self._euclidean_distance(point, query_point) <= self.epsilon:
                neighbors.append(i)
        return neighbors

    def _density_connected(self, points_neighbours, point_index, cluster_label, labels):
        neighbours = points_neighbours[point_index]
        for neighbour_point_index in neighbours:
            if labels[neighbour_point_index] == -1:
                labels[neighbour_point_index] = cluster_label
                if len(points_neighbours[neighbour_point_index]) >= self.minpts:
                    self._density_connected(points_neighbours, neighbour_point_index, cluster_label, labels)

    def fit_predict(self, X):
        labels = np.full(len(X), -1)
        core_points_indexes = []
        points_neighbours = []
        for i, point in enumerate(X):
            neighbours = self.get_neighbors(X, point)
            points_neighbours.append(neighbours)
            if len(neighbours) >= self.minpts:
                core_points_indexes.append(i)

        cluster_label = 0
        for point_index in core_points_indexes:
            if labels[point_index] == -1:
                labels[point_index] = cluster_label
                self._density_connected(points_neighbours, point_index, cluster_label, labels)
                cluster_label += 1

        return labels


if __name__ == "__main__":
    X = np.array([
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

    dbscan = DBSCAN(epsilon=1, minpts=2)
    labels = dbscan.fit_predict(X)
    plot_clusters(X, labels, "Clustering")
    print(labels)

    dbscan_builtin = DBSCANB(eps=1, min_samples=2)
    dbscan_builtin.fit(X)
    labels = dbscan_builtin.labels_
    print(labels)
