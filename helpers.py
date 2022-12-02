import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    print(all_files)
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


def preprocessing(data):

    df = pd.DataFrame(data)
    #mean=df.mean(axis=1)
    median = df.median(axis=1)
    #change Nan value to the mean value of the column
    #df.fillna(median)
    #remove outliers: remplace by median values or +/- 3x the standard deviation of the feature
    #df.std(ddof=0) divide by N instead of N-1
    df[df > 3 * df.std() ] = 3 * df.std()
    df[df < 3 * df.std()] = 3 * df.std()

    #find low variance from visualisation, remove? df.drop([index],axis=1)

    #standardize # define standard scaler
    scaler = StandardScaler()
    # transform data -> return an array of the standardize values
    data = scaler.fit_transform(df)

    return data


def visualisation (data):
    df = pd.DataFrame(data)
    df.describe(include='all')
    #want to show column with low variance
    # df.std(ddof=0) divide by N instead of N-1
    variance = df.std()
    df = df.reset_index()  # make sure indexes pair with number of rows

    for index, row in variance.iterrows():
        if row[index] < 0.001:
            df.append(variance[index])


    #want to see how it plots x vs y ?
    #df.plot()
