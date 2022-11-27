import glob
import os
import numpy as np
from pathlib import Path
import scipy.io


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
    # change to use the IDs as labels
    y_labels = np.arange(1, 49, 1)

    return final_matrix, y_labels


def load_csv_data_updated(data_path):
    path = os.path.join(data_path, '**/*.mat')
    files = glob.glob(path, recursive=True)

    final_data_set_train_wl2 = np.array([])
    final_data_set_test_wl2 = np.array([])
    final_data_set_train_wlog = np.array([])
    final_data_set_test_wlog = np.array([])
    # TODO: somewhere we will need to add the labels !!!!

    for file in files:
        number_of_files = 0
        # TODO:  we will have to put all of this in a function that we can call for both cases of the if

        data = scipy.io.loadmat(file, appendmat=0)
        results = data['RESULTS']
        session = results["session"]
        alphas = results["alphas"]
        betas = results["betas"]
        id = results["ID"]
        wl2 = results["W_l2"]
        wlog = results["W_log"]
        wl2 = wl2[0][0]
        wlog = wlog[0][0]
        alphas = alphas[0][0].T
        betas = betas[0][0].T

        if session == "3-Restin_rmegpreproc_bandpass-envelop":
            # train for wl2
            for i in range(alphas.shape[0]):
                data_set_wl2 = np.array([])
                for j in range(wl2.shape[3]-1):
                    matrix = wl2[:, :, i, j]
                    column_vector = read_matrix(matrix)
                    if data_set_wl2.size == 0:
                        data_set_wl2 = column_vector
                    else:
                        data_set_wl2 = np.vstack((data_set_wl2, column_vector))

                if final_data_set_train_wl2.size == 0:
                    final_data_set_train_wl2 = data_set_wl2
                else:
                    final_data_set_train_wl2[:, :, i] = np.vstack((final_data_set_train_wl2[:, :, i], data_set_wl2))

            # train for wl2
            for a in range(alphas.shape[0]):
                data_set_wlog = np.array([])
                for b in range(wlog.shape[3] - 1):
                    matrix = wlog[:, :, a, b]
                    column_vector = read_matrix(matrix)
                    if data_set_wlog.size == 0:
                        data_set_wlog = column_vector
                    else:
                        data_set_wlog = np.vstack((data_set_wlog, column_vector))

                if final_data_set_train_wlog.size == 0:
                    final_data_set_train_wlog = data_set_wlog
                    final_data_set_train_wlog = np.expand_dims(final_data_set_train_wlog, 2)
                else:
                    final_data_set_train_wlog[:, :, a] = np.vstack(
                        (final_data_set_train_wlog[:, :, a], data_set_wlog))

        elif session == "4-Restin_rmegpreproc_bandpass-envelop":
            # train for wl2
            for i in range(alphas.shape[0]):
                data_set_wl2 = np.array([])
                for j in range(wl2.shape[3] - 1):
                    matrix = wl2[:, :, i, j]
                    column_vector = read_matrix(matrix)
                    if data_set_wl2.size == 0:
                        data_set_wl2 = column_vector
                    else:
                        data_set_wl2 = np.vstack((data_set_wl2, column_vector))

                if final_data_set_test_wl2.size == 0:
                    final_data_set_test_wl2 = data_set_wl2
                    final_data_set_test_wl2 = np.expand_dims(final_data_set_test_wl2, 2)
                else:
                    final_data_set_test_wl2[:, :, i] = np.vstack((final_data_set_test_wl2[:, :, i], data_set_wl2))

            # train for wl2
            for a in range(alphas.shape[0]):
                data_set_wlog = np.array([])
                for b in range(wlog.shape[3] - 1):
                    matrix = wlog[:, :, a, b]
                    column_vector = read_matrix(matrix)
                    if data_set_wlog.size == 0:
                        data_set_wlog = column_vector
                    else:
                        data_set_wlog = np.vstack((data_set_wlog, column_vector))

                if final_data_set_test_wlog.size == 0:
                    final_data_set_test_wlog = data_set_wlog
                    final_data_set_test_wlog = np.expand_dims(final_data_set_test_wlog, 2)
                else:
                    final_data_set_test_wlog[:, :, a] = np.vstack(
                        (final_data_set_test_wlog[:, :, a], data_set_wlog))

        number_of_files +=1
    '''
    for alpha in range(20):
        np.savetxt('train_wl2_' + str(alpha), final_data_set_train_wl2[:, :, alpha], delimeter=' ')
        np.savetxt('test_wl2_' + str(alpha), final_data_set_test_wl2[:, :, alpha], delimeter=' ')
        
    for beta in range(20):
        np.savetxt('train_wlog_' + str(beta), final_data_set_train_wlog[:, :, beta], delimeter=' ')
        np.savetxt('test_wlog_' + str(beta), final_data_set_test_wlog[:, :, beta], delimeter=' ')
        
    '''
    return final_data_set_train_wl2, final_data_set_train_wlog, final_data_set_test_wlog, final_data_set_test_wl2
def read_matrix(matrix, size_of_matrix=148):
    # this function works!
    x = np.reshape(matrix, (size_of_matrix, size_of_matrix))
    indices_upper_triangular = np.triu_indices(size_of_matrix, 1)
    upper_triangular_matrix_vector = x[indices_upper_triangular]

    # I do not understand why but the next line doesn't change anything,
    # so the return is a column vector of shape (10'878, )
    upper_triangular_matrix_vector = upper_triangular_matrix_vector.reshape(1, -1)

    return upper_triangular_matrix_vector


def readfile(filename, size_of_matrix=148):
    x = np.genfromtxt(filename)
    x = np.reshape(x, (size_of_matrix, size_of_matrix))

    indices_upper_triangular = np.triu_indices(size_of_matrix, 1)
    upper_triangular_matrix = x[indices_upper_triangular]

    return upper_triangular_matrix
