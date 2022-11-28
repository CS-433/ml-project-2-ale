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
    # y_labels = np.arange(1, 49, 1)
    y_labels = np.arange(1, 4, 1)

    return final_matrix, y_labels


def load_csv_data_updated(data_path):
    path = os.path.join(data_path, '**/*.mat')
    files = glob.glob(path, recursive=True)

    final_data_set_train_wl2 = []
    final_data_set_test_wl2 = []
    final_data_set_train_wlog = []
    final_data_set_test_wlog = []

    # TODO: somewhere we will need to add the labels !!!!
    number_of_files = 0
    for file in files:

        # TODO:  we will have to put all of this in a function that we can call for both cases of the if

        data = scipy.io.loadmat(file, appendmat=0)
        results = data['RESULTS']
        session = results["session"]
        alphas = results["alphas"]
        betas = results["betas"]
        id = results["ID"]
        id = np.squeeze(id)
        wl2 = results["W_l2"]
        wlog = results["W_log"]
        size_wl2 = results["W_l2_size"]
        size_wl2 = size_wl2[0][0]
        size_wl2 = np.squeeze(size_wl2)
        size_wlog = results["W_log_size"]
        size_wlog = size_wlog[0][0]
        size_wlog = np.squeeze(size_wlog)
        wl2 = wl2[0][0]
        wlog = wlog[0][0]
        alphas = alphas[0][0].T
        betas = betas[0][0].T
        # print(size_wl2)
        # print(size_wlog)

        if session == "3-Restin_rmegpreproc_bandpass-envelop":
            # train for wl2
            for i in range(alphas.shape[0]):
                data_set_wl2 = np.array([])
                # range(wl2.shape[3]-1) --> this doesn't work, j = 23 or 22 sometimes, idk why...
                for j in range(size_wl2[3]-1):
                    matrix = wl2[:, :, i, j]
                    column_vector = read_matrix(matrix)
                    column_vector = np.insert(column_vector, 0, id)
                    if data_set_wl2.size == 0:
                        data_set_wl2 = column_vector
                    else:
                        data_set_wl2 = np.vstack((data_set_wl2, column_vector))

                if number_of_files == 0:
                    final_data_set_train_wl2.append(data_set_wl2)
                else:
                    final_data_set_train_wl2[i] = np.vstack((final_data_set_train_wl2[i], data_set_wl2))

            # train for wlog
            for a in range(betas.shape[0]):
                data_set_wlog = np.array([])
                # range(wlog.shape[3]-1)
                for b in range(size_wlog[3]-1):
                    matrix = wlog[:, :, a, b]
                    column_vector = read_matrix(matrix)
                    column_vector = np.insert(column_vector, 0, id)
                    if data_set_wlog.size == 0:
                        data_set_wlog = column_vector
                    else:
                        data_set_wlog = np.vstack((data_set_wlog, column_vector))

                if number_of_files == 0:
                    final_data_set_train_wlog.append(data_set_wlog)
                else:
                    final_data_set_train_wlog[a] = np.vstack((final_data_set_train_wlog[a], data_set_wlog))

        elif session == "4-Restin_rmegpreproc_bandpass-envelop":
            # test for wl2
            for i in range(alphas.shape[0]):
                data_set_wl2 = np.array([])
                # range(wl2.shape[3] - 1) -> doesn't work idk why
                # range(24) -> index 23 is out of bounds for axis 3 with size 23???
                for j in range(size_wl2[3]-1):
                    matrix = wl2[:, :, i, j]
                    column_vector = read_matrix(matrix)
                    column_vector = np.insert(column_vector, 0, id)
                    if data_set_wl2.size == 0:
                        data_set_wl2 = column_vector
                    else:
                        data_set_wl2 = np.vstack((data_set_wl2, column_vector))

                if number_of_files == 5:
                    final_data_set_test_wl2.append(data_set_wl2)
                else:
                    final_data_set_test_wl2[i] = np.vstack((final_data_set_test_wl2[i], data_set_wl2))

            # test for wlog
            for a in range(betas.shape[0]):
                data_set_wlog = np.array([])
                for b in range(size_wlog[3]-1):
                    matrix = wlog[:, :, a, b]
                    column_vector = read_matrix(matrix)
                    column_vector = np.insert(column_vector, 0, id)
                    if data_set_wlog.size == 0:
                        data_set_wlog = column_vector
                    else:
                        data_set_wlog = np.vstack((data_set_wlog, column_vector))

                if number_of_files == 5:
                    final_data_set_test_wlog.append(data_set_wlog)
                else:
                    final_data_set_test_wlog[a] = np.vstack((final_data_set_test_wlog[a], data_set_wlog))

        number_of_files += 1

    for alpha in range(20):
        np.savetxt(r'data_sets/train_wl2_' + str(alpha), final_data_set_train_wl2[alpha], delimiter=' ')
        np.savetxt(r'data_sets/test_wl2_' + str(alpha), final_data_set_test_wl2[alpha], delimiter=' ')

    for beta in range(20):
        np.savetxt(r'data_sets/train_wlog_' + str(beta), final_data_set_train_wlog[beta], delimiter=' ')
        np.savetxt(r'data_sets/test_wlog_' + str(beta), final_data_set_test_wlog[beta], delimiter=' ')

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


'''
def load_csv_data_updated(data_path):
    path = os.path.join(data_path, '**/*.mat')
    files = glob.glob(path, recursive=True)

    final_data_set_train_wl2 = np.empty((75, 10878, 20))
    final_data_set_test_wl2 = np.empty((75, 10878, 20))
    final_data_set_train_wlog = np.empty((75, 10878, 20))
    final_data_set_test_wlog = np.empty((75, 10878, 20))
    # TODO: somewhere we will need to add the labels !!!!

    for file in files:
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
            # data_sets_wl2 = []
            for i in range(alphas.shape[0]):
                data_set_wl2 = np.empty((25, 10878))
                for j in range(wl2.shape[3]):
                    matrix = wl2[:, :, i, j]
                    column_vector = read_matrix(matrix)
                    np.vstack((data_set_wl2, column_vector))
                    print(data_set_wl2.shape)
                np.vstack((final_data_set_train_wl2[:, :, i], data_set_wl2))

            # stacks the data set in 3rd dimension
            # data_sets_wl2 = np.dstack(data_sets_wl2)

            # final_data_set_train_wl2.append(data_sets_wl2)

            # train for wlog
            data_sets_wlog = []
            for a in range(betas.shape[0]):
                data_set_wlog = []
                for b in range(wlog.shape[3]):
                    matrix = wlog[:, :, a, b]
                    column_vector = read_matrix(matrix)
                    data_set_wlog.append(column_vector)
                data_sets_wlog.append(data_set_wlog)

            data_sets_wlog = np.dstack(data_sets_wlog)  # stacks the data set in 3rd dimension
            final_data_set_train_wlog.append(data_sets_wlog)

        elif session == "4-Restin_rmegpreproc_bandpass-envelop":
            # test for wl2
            data_sets_wl2 = []
            for i in range(alphas.shape[0]):
                data_set_wl2 = []
                for j in range(wl2.shape[3]):
                    matrix = wl2[:, :, i, j]
                    column_vector = read_matrix(matrix)
                    data_set_wl2.append(column_vector)
                data_sets_wl2.append(data_set_wl2)

            data_sets_wl2 = np.dstack(data_sets_wl2)  # stacks the data set in 3rd dimension
            final_data_set_test_wl2.append(data_sets_wl2)

            # train for wlog
            data_sets_wlog = []
            for a in range(betas.shape[0]):
                data_set_wlog = []
                for b in range(wlog.shape[3]):
                    matrix = wlog[:, :, a, b]
                    column_vector = read_matrix(matrix)
                    data_set_wlog.append(column_vector)
                data_sets_wlog.append(data_set_wlog)

            data_sets_wlog = np.dstack(data_sets_wlog)  # stacks the data set in 3rd dimension
            final_data_set_test_wlog.append(data_sets_wlog)

    return final_data_set_train_wl2, final_data_set_train_wlog, final_data_set_test_wlog, final_data_set_test_wl2
'''
