
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
    print(all_files)
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


def load_mat_file(filename):
    """
        loads a .mat file containing a structure RESULTS containing the matrix W generated from a learning graph script
        RESULTS contains the matrix W for each regularization (l2 and log) of size:
            brain regions x brain regions x alphas/betas (hyper-params of the learning graph algorithm) x epochs
        RESULTS also contains the other information on the graph learned such as: the patient's ID,
        the number of the session, the frequency band measured, the distance used in the algorithm and
        the signal format (envelop or bandpass)
    :param filename: name of the file to unpack
    :return: session, alphas, betas, id, frequency band, size of wl2, size of wlog, wl2, wlog
    """
    # load the file
    data = scipy.io.loadmat(filename, appendmat=0)
    results = data['RESULTS']

    # unpack the variables needed from the results dictionary
    # session
    session = results["session"]
    # alphas, betas
    alphas = results["alphas"]
    betas = results["betas"]
    alphas = alphas[0][0].T
    betas = betas[0][0].T
    # id
    id_ = results["ID"]
    # frequency band
    frequency_band = results["band"]
    frequency_band = frequency_band[0][0]
    # size of the data
    size_wl2 = results["W_l2_size"]
    size_wl2 = size_wl2[0][0]
    size_wlog = results["W_log_size"]
    size_wlog = size_wlog[0][0]
    # main data matrix
    wl2 = results["W_l2"]
    wl2 = wl2[0][0]
    wlog = results["W_log"]
    wlog = wlog[0][0]

    return session, alphas, betas, np.squeeze(id_), np.squeeze(frequency_band), np.squeeze(size_wl2), \
           np.squeeze(size_wlog), wl2, wlog


def create_set_epochs(data_set, w_sub_matrix, id):
    column_vector = read_matrix(w_sub_matrix)
    column_vector = np.insert(column_vector, 0, id)
    if data_set.size == 0:
        data_set = column_vector
    else:
        data_set = np.vstack((data_set, column_vector))
    return data_set


def complete_final_data_set(final_data_set, temp_data_set, params, col_index):
    if len(final_data_set) < params.shape[0]:
        final_data_set.append(temp_data_set)
    else:
        final_data_set[col_index] = np.vstack((final_data_set[col_index], temp_data_set))

    return final_data_set


def create_final_data_set(data_set_alpha, data_set_beta, data_set_delta, data_set_gamma, data_set_theta, params, size_w,
                          w, id, frequency_band, last_epoch):
    for i in range(params.shape[0]):
        data_set_temp = np.array([])
        if last_epoch:
            matrix = w[:, :, i, size_w[3] - 1]
            data_set_temp = create_set_epochs(data_set_temp, matrix, id)
        else:
            for j in range(size_w[3] - 1):
                matrix = w[:, :, i, j]
                data_set_temp = create_set_epochs(data_set_temp, matrix, id)
        if frequency_band == "alpha":
            data_set_alpha = complete_final_data_set(data_set_alpha, data_set_temp, params, i)
        elif frequency_band == "beta":
            data_set_beta = complete_final_data_set(data_set_beta, data_set_temp, params, i)
        elif frequency_band == "delta":
            data_set_delta = complete_final_data_set(data_set_delta, data_set_temp, params, i)
        elif frequency_band == "gamma":
            data_set_gamma = complete_final_data_set(data_set_gamma, data_set_temp, params, i)
        elif frequency_band == "theta":
            data_set_theta = complete_final_data_set(data_set_theta, data_set_temp, params, i)
    return data_set_alpha, data_set_beta, data_set_delta, data_set_gamma, data_set_theta


def create_all_sets(data_path, save_file=True, verbose=True, last_epoch=False):
    path = os.path.join(data_path, '**/*.mat')
    files = glob.glob(path, recursive=True)

    # declare all the final data sets (1 per band per regularization) as an empty list
    final_data_set_train_wl2_alpha = []
    final_data_set_train_wl2_beta = []
    final_data_set_train_wl2_delta = []
    final_data_set_train_wl2_gamma = []
    final_data_set_train_wl2_theta = []
    final_data_set_test_wl2_alpha = []
    final_data_set_test_wl2_beta = []
    final_data_set_test_wl2_delta = []
    final_data_set_test_wl2_gamma = []
    final_data_set_test_wl2_theta = []
    final_data_set_train_wlog_alpha = []
    final_data_set_train_wlog_beta = []
    final_data_set_train_wlog_delta = []
    final_data_set_train_wlog_gamma = []
    final_data_set_train_wlog_theta = []
    final_data_set_test_wlog_alpha = []
    final_data_set_test_wlog_beta = []
    final_data_set_test_wlog_delta = []
    final_data_set_test_wlog_gamma = []
    final_data_set_test_wlog_theta = []

    number_of_files = 0
    for file in files:
        # load all the information needed and the data from the .mat file generated by the learning graph script
        session, alphas, betas, id, frequency_band, size_wl2, size_wlog, wl2, wlog = load_mat_file(file)

        # create all the train data set with session 3
        if session == "3-Restin_rmegpreproc_bandpass-envelop":
            # train for wl2
            final_data_set_train_wl2_alpha, final_data_set_train_wl2_beta, \
                final_data_set_train_wl2_delta, final_data_set_train_wl2_gamma, \
                final_data_set_train_wl2_theta = create_final_data_set(final_data_set_train_wl2_alpha,
                                                                       final_data_set_train_wl2_beta,
                                                                       final_data_set_train_wl2_delta,
                                                                       final_data_set_train_wl2_gamma,
                                                                       final_data_set_train_wl2_theta, alphas, size_wl2,
                                                                       wl2, id, frequency_band, last_epoch)

            # train for wlog
            final_data_set_train_wlog_alpha, final_data_set_train_wlog_beta, \
                final_data_set_train_wlog_delta, final_data_set_train_wlog_gamma, \
                final_data_set_train_wlog_theta = create_final_data_set(final_data_set_train_wlog_alpha,
                                                                        final_data_set_train_wlog_beta,
                                                                        final_data_set_train_wlog_delta,
                                                                        final_data_set_train_wlog_gamma,
                                                                        final_data_set_train_wlog_theta, betas,
                                                                        size_wlog, wlog, id, frequency_band, last_epoch)

        # create the test data set with session 4
        elif session == "4-Restin_rmegpreproc_bandpass-envelop":
            # test for wl2
            final_data_set_test_wl2_alpha, final_data_set_test_wl2_beta, \
                final_data_set_test_wl2_delta, final_data_set_test_wl2_gamma, \
                final_data_set_test_wl2_theta = create_final_data_set(final_data_set_test_wl2_alpha,
                                                                      final_data_set_test_wl2_beta,
                                                                      final_data_set_test_wl2_delta,
                                                                      final_data_set_test_wl2_gamma,
                                                                      final_data_set_test_wl2_theta, alphas, size_wl2,
                                                                      wl2, id, frequency_band, last_epoch)

            # test for wlog
            final_data_set_test_wlog_alpha, final_data_set_test_wlog_beta, \
                final_data_set_test_wlog_delta, final_data_set_test_wlog_gamma, \
                final_data_set_test_wlog_theta = create_final_data_set(final_data_set_test_wlog_alpha,
                                                                       final_data_set_test_wlog_beta,
                                                                       final_data_set_test_wlog_delta,
                                                                       final_data_set_test_wlog_gamma,
                                                                       final_data_set_test_wlog_theta, betas,
                                                                       size_wlog, wlog, id, frequency_band, last_epoch)

        if verbose:
            print("file " + str(number_of_files) + " loaded")
            number_of_files += 1

    # save all the files
    if save_file:
        if last_epoch:
            name = '_all_epochs_combined'
        else:
            name = ''

        for alpha in range(alphas.shape[0]):
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wl2_' + 'alpha_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_train_wl2_alpha[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wl2_' + 'beta_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_train_wl2_beta[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wl2_' + 'delta_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_train_wl2_delta[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wl2_' + 'gamma_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_train_wl2_gamma[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wl2_' + 'theta_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_train_wl2_theta[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wl2_' + 'alpha_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_test_wl2_alpha[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wl2_' + 'beta_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_test_wl2_beta[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wl2_' + 'delta_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_test_wl2_delta[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wl2_' + 'gamma_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_test_wl2_gamma[alpha], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wl2_' + 'theta_' + str(
                np.squeeze(alphas[alpha])) + name + '.txt', final_data_set_test_wl2_theta[alpha], delimiter=' ')
            if verbose:
                print("all files for alpha = " + str(np.squeeze(alphas[alpha])) + " saved")

        for beta in range(betas.shape[0]):
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wlog_' + 'alpha_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_train_wlog_alpha[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wlog_' + 'beta_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_train_wlog_beta[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wlog_' + 'delta_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_train_wlog_delta[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wlog_' + 'gamma_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_train_wlog_gamma[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/train/train_wlog_' + 'theta_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_train_wlog_theta[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wlog_' + 'alpha_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_test_wlog_alpha[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wlog_' + 'beta_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_test_wlog_beta[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wlog_' + 'delta_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_test_wlog_delta[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wlog_' + 'gamma_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_test_wlog_gamma[beta], delimiter=' ')
            np.savetxt(r'../../Data3/Hamid_ML4Science_ALE/data_sets/test/test_wlog_' + 'theta_' + str(
                np.squeeze(betas[beta])) + name + '.txt', final_data_set_test_wlog_theta[beta], delimiter=' ')
            if verbose:
                print("all files for beta = " + str(np.squeeze(betas[beta])) + " saved")

    return final_data_set_train_wl2_alpha, final_data_set_train_wl2_beta, final_data_set_train_wl2_delta, \
        final_data_set_train_wl2_gamma, final_data_set_train_wl2_theta, final_data_set_test_wl2_alpha, \
        final_data_set_test_wl2_beta, final_data_set_test_wl2_delta, final_data_set_test_wl2_gamma, \
        final_data_set_test_wl2_theta, final_data_set_train_wlog_alpha, final_data_set_train_wlog_beta, \
        final_data_set_train_wlog_delta, final_data_set_train_wlog_gamma, final_data_set_train_wlog_theta, \
        final_data_set_test_wlog_alpha, final_data_set_test_wlog_beta, final_data_set_test_wlog_delta, \
        final_data_set_test_wlog_gamma, final_data_set_test_wlog_theta


def read_matrix(matrix, size_of_matrix=148):
    x = np.reshape(matrix, (size_of_matrix, size_of_matrix))
    indices_upper_triangular = np.triu_indices(size_of_matrix, 1)
    upper_triangular_matrix_vector = x[indices_upper_triangular]
    upper_triangular_matrix_vector = upper_triangular_matrix_vector.reshape(1, -1)

    return upper_triangular_matrix_vector


def readfile(filename, size_of_matrix=148):
    x = np.genfromtxt(filename)
    x = np.reshape(x, (size_of_matrix, size_of_matrix))

    indices_upper_triangular = np.triu_indices(size_of_matrix, 1)
    upper_triangular_matrix = x[indices_upper_triangular]

    return upper_triangular_matrix


'''
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
    
'''

def load_data_set(band, regularization, type, parameter, epochs_combined=False, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/'):

    """
        Loads the data sets created by create_sets with certain parameters and returns it.

        Arguments:
            - band: string that represents which band should be selected among 5 possibilities
                - alpha
                - beta
                - delta
                - gamma
                - theta
            - regularization: string either wl2 or wlog specifying which norm should be used
            - type: string, either test or train specifying which type of data set we want
            - parameter: float that represents either the parameter value of the alpha value if l2 is chosen or
                         beta if log is chosen. 20 possible value
            - path: string where the test and train files are, default value is the path on the servers

        Returns:
            - the file corresponding to those parameters
        """

    path_of_file = path
    if type == 'train':
        path_of_file += 'train/'
        if regularization == 'wlog':
            path_of_file += 'train_wlog_'
        elif regularization == 'wl2':
            path_of_file += 'train_wl2_'
    elif type == 'test':
        path_of_file += 'test/'
        if regularization == 'wlog':
            path_of_file += 'test_wlog_'
        elif regularization == 'wl2':
            path_of_file += 'test_wl2_'

    if band == 'alpha':
        path_of_file += 'alpha_'
    elif band == 'beta':
        path_of_file += 'beta_'
    elif band == 'delta':
        path_of_file += 'delta_'
    elif band == 'gamma':
        path_of_file += 'gamma_'
    elif band == 'theta':
        path_of_file += 'theta_'

    if epochs_combined:
        path_of_file = path_of_file + str(parameter) + '_all_epochs_combined' + '.txt'
    else:
        path_of_file = path_of_file + str(parameter) + '.txt'

    matrix = np.loadtxt(path_of_file)

    ids = matrix[:, 0]
    matrix = np.delete(matrix, 0, axis=1)

    return matrix, ids

