import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path


def load_csv_data(data_path):
    """
    Loads data and returns final matrix with the features.

    Arguments:
        - data_path: the path of the data files in .csv

    Returns:
        - final_matrix
        - y_labels
    """
    path = data_path  # use my path
    all_files = Path(path).glob('*.csv')
    # all_files = glob.glob(os.path.join(path, "/*.csv"))

    final_matrix = []
    for filename in all_files:
        line = readfile(filename)
        final_matrix.append(line)

    final_matrix = np.array(final_matrix)
    y_labels = np.arange(1, 49, 1)

    return final_matrix, y_labels


def readfile(filename, size_of_matrix=148):
    x = np.genfromtxt(filename)
    x = np.reshape(x, (size_of_matrix, size_of_matrix))

    indices_upper_triangular = np.triu_indices(size_of_matrix, 1)
    upper_triangular_matrix = x[indices_upper_triangular]

    return upper_triangular_matrix
