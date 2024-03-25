import csv
import pandas as pd
import timeit

import numpy as np
from sklearn.decomposition import PCA


def read_2d_array_from_file(filename):
    return pd.read_csv(filename, header=None).values


def extract_data_matrix():
    activity = 19
    person = 8
    chunk = 60
    cnt = 0
    data = np.empty((activity * person * chunk, 125, 45))
    for i in range(1, activity + 1):
        for j in range(1, person + 1):
            for k in range(1, chunk + 1):
                i_str = str(i).zfill(2)
                k_str = str(k).zfill(2)
                f = f"data/a{i_str}/p{j}/s{k_str}.txt"
                sample = read_2d_array_from_file(f)
                data[cnt] = sample
                cnt += 1
    return data


def reduce_data_using_mean_for_matrix(big_data):
    data = np.array([])
    for matrix in big_data:
        mean_matrix = np.mean(matrix, axis=0)
        data = np.vstack([data, mean_matrix]) if data.size else mean_matrix
    return data


def reduce_data_using_flatten_for_matrix(big_data):
    flattened_matrices = [np.reshape(matrix, -1) for matrix in big_data]
    return np.vstack(flattened_matrices)


def split_data(data):
    total_splits = len(data) // 60
    reduced_train_data = np.array([])
    reduced_test_data = np.array([])
    train_data_labels = np.array([])
    test_data_labels = np.array([])
    cnt = 0
    for i in range(total_splits):
        if i % 8 == 0:
            cnt += 1
        start_index = i * 60
        end_index = start_index + 60
        train_data = data[start_index:start_index + 48]
        test_data = data[start_index + 48:end_index]
        train_data_labels = np.append(train_data_labels, [cnt] * len(train_data))
        test_data_labels = np.append(test_data_labels, [cnt] * len(test_data))
        reduced_train_data = np.vstack([reduced_train_data, train_data]) if reduced_train_data.size else train_data
        reduced_test_data = np.vstack([reduced_test_data, test_data]) if reduced_test_data.size else test_data

    return reduced_train_data, reduced_test_data, train_data_labels, test_data_labels


def get_data_for_solution_one():
    big_data = extract_data_matrix()
    data = reduce_data_using_mean_for_matrix(big_data)
    return split_data(data)


def get_data_for_solution_two():
    big_data = extract_data_matrix()
    data = reduce_data_using_flatten_for_matrix(big_data)
    pca = PCA(n_components=0.8)
    reduced_data = pca.fit_transform(data)
    print(f"Number Of Components : {pca.n_components_}")
    return split_data(reduced_data)


if __name__ == '__main__':
    get_data_for_solution_one()
    get_data_for_solution_two()
