from extractData import *
from evaluation import Evaluation
from k_means import KMeans


import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score

def compute_testing_clusters(testing_data, centroids):
    clusters = np.argmin(np.linalg.norm(testing_data[:, None] - centroids, axis=2), axis=1)
    return clusters


def apply_kmeans(training_data, test_data, test_labels, k_values):
    for k in k_values:
        print("K:", k)
        kmeans = KMeans(k=k)
        _, centroids = kmeans.fit_predict(training_data)
        testing_clusters = compute_testing_clusters(test_data, centroids)
        evaluation = Evaluation(testing_clusters, test_labels)
        evaluation.evaluate()
        print("---------------------------------------------------")


if __name__ == "__main__":
    training_data, test_data, tL, test_labels = get_data_for_solution_one()
    # Assuming you have loaded your data into 'data' and 'data_labels'

    # Apply KMeans
    kmeans = KMeans(n_clusters=19)
    kmeans.fit(training_data)
    predicted_labels = kmeans.labels_

    # Evaluate with F1 Score
    f1 = f1_score(tL, predicted_labels, average='none')
    print("F1 Score:", f1)
    print(f1)
    k_values = [8, 13, 19, 28, 38]
    print("Solution One On Mean Data")
    apply_kmeans(training_data, test_data, test_labels, k_values)
    training_data, test_data, _, test_labels = get_data_for_solution_two()
    print("Solution Two On Flattened Data And PCA")
    apply_kmeans(training_data, test_data, test_labels, k_values)


