import numpy as np

import numpy as np


class Evaluation:
    def __init__(self, clustering_result, labels):
        self.clustering_result = clustering_result
        self.labels = labels
        self.clusters = None

    def separate_clusters_on_result(self):
        self.clusters = {}
        for i, cluster_id in enumerate(self.clustering_result):
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(i)
        return self.clusters

    def evaluate(self):
        self.separate_clusters_on_result()
        print("Precision: ")
        print(self._compute_precision())
        print("Recall: ")
        print(self._compute_recall())
        print("F1 Score: ")
        print(self._compute_f1_score())
        print("Conditional Entropy: ")
        print(self._compute_conditional_entropy())

    def _compute_precision(self):
        total_precision = 0
        total_len = self.labels.shape[0]
        for cluster in self.clusters:
            labels_in_cluster = self.labels[self.clusters[cluster]]
            unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)
            max_count = np.max(counts)
            total_precision += max_count
        precision = total_precision / total_len
        return precision

    def _compute_recall(self):
        total_recall = 0
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        for cluster in self.clusters:
            labels_in_cluster = self.labels[self.clusters[cluster]]
            unique_labels_in_cluster, counts_in_cluster = np.unique(labels_in_cluster, return_counts=True)
            max_count = np.argmax(counts_in_cluster)
            max_label=unique_labels_in_cluster[max_count]
            total_count=counts[np.where(unique_labels == max_label)[0][0]]
            temp = counts_in_cluster[max_count] / total_count
            total_recall += temp
        return total_recall

    def _compute_f1_score(self):
        result = 0
        unique_labels_total, counts_total = np.unique(self.labels, return_counts=True)
        for cluster in self.clusters:
            labels_in_cluster = self.labels[self.clusters[cluster]]
            unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)
            max_count = np.max(counts)
            precision = max_count / len(self.clusters[cluster])
            max_label=unique_labels[np.argmax(counts)]
            total_count = counts_total[np.where(unique_labels_total == max_label)[0][0]]
            recall = max_count / total_count
            result += 2 * precision * recall / (precision + recall)
        result /= len(self.clusters)
        return result

    def _compute_conditional_entropy(self):
        result = 0
        for cluster in self.clusters:
            labels_in_cluster = self.labels[self.clusters[cluster]]
            unique_labels, counts = np.unique(labels_in_cluster, return_counts=True)
            cluster_result = 0
            for label in unique_labels:
                temp = counts[np.where(unique_labels == label)[0][0]] / len(self.clusters[cluster])
                cluster_result -= temp * np.log2(temp)

            result += cluster_result * len(self.clusters[cluster]) / len(self.labels)
        return result

if __name__ == '__main__':
    clustering_result = np.array([0,0,0,0,0,0,1,1,1,1,1,1,1,2,2,2,2,2])
    labels = np.array([0,0,0,0,0,1,0,1,1,1,1,2,2,1,1,2,2,2])
    # clustering_result = np.array([0, 1, 2, 3, 4])
    # labels = np.array([0, 0, 0, 0, 0])
    evaluation = Evaluation(clustering_result, labels)
    evaluation.evaluate()
