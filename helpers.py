import pandas as pd
import glob
import os
import numpy as np
from pathlib import Path
import scipy.io


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
    """
    Completes a given data set by adding a row with the vectorized upper triangular part of a given matrix as
    features and given id as label. The data set can be empty, in this case it will be initialized with the row
    described above

    :param data_set: data set to be completed size (N,D)
    :param w_sub_matrix: ndarray of size (D,D) or empty
    :param id: ID of the patient corresponding to the W matrix
    :return: The data set with a row added
    """
    # vectorize the sub_matrix
    column_vector = read_matrix(w_sub_matrix)
    # add the ID at the begining
    column_vector = np.insert(column_vector, 0, id)
    # completes the dataset
    if data_set.size == 0:
        data_set = column_vector
    else:
        data_set = np.vstack((data_set, column_vector))
    return data_set


def complete_final_data_set(final_data_set, temp_data_set, params, col_index):
    """
    Get a dataset from the final_data_set list at col_index and stacks the temp_data_set under it. The aim is to
    have one data set per hyperparameter (params), one sample of each data set being one epoch from one patient.
    If the final_data_set list is not yet of the right length (length of param), the temporary dataset is simply
    appended at the end of the list

    :param final_data_set: a list of size M containing data sets of sizes (N,D)
    :param temp_data_set: data set of size (n,D) to stack under a dataset of the list M. In our setting, each row of this
    dataset is a sample (i.e. ID of a person followed by features) and the temp_data_set contains all the epochs samples
    from only one person
    :param params: Hyper parameter of the learning graph, used to know the size of the final_data_set list
    :param col_index: indicated under which element of the final_data_set list temp_data_set should be stacked
    :return: The final_data_set list now completed with temp_data_set
    """
    # if the list of data set is not yet long enough -> no data set exists yet for this hyperparameter so the temporary
    # data set is appended at its end
    if len(final_data_set) < params.shape[0]:
        final_data_set.append(temp_data_set)
    # else it is simply stacked under the existing one at the specified position in the list
    else:
        final_data_set[col_index] = np.vstack((final_data_set[col_index], temp_data_set))

    return final_data_set


def create_final_data_set(data_set_alpha, data_set_beta, data_set_delta, data_set_gamma, data_set_theta, params, size_w,
                          w, id, frequency_band, last_epoch):
    """
    Creates a complete list for the frequency band passed in argument from the adjacency matrices returned by the
    learning graph script. The list is of the length of params and contains one dataset per hyperparameter (param).
    One sample of each dataset (list elements) is one epoch from one patient. For the other frequency bands, the
    lists are returned unchanged

    :param data_set_alpha: empty list in which to store all the datasets for frequency band alpha
    :param data_set_beta: empty list in which to store all the datasets for frequency band beta
    :param data_set_delta: empty list in which to store all the datasets for frequency band delta
    :param data_set_gamma: empty list in which to store all the datasets for frequency band gamma
    :param data_set_theta: empty list in which to store all the datasets for frequency band theta
    :param params: ndarray of size (p,). Hyperparameters of the learning graph, W contains 1 matrix per hyperparameter
    :param size_w: scalar, size of the matrix W
    :param w: Adjacency matrices created by the learning graph algorithm. ndarray of size (b, b, p, e+1).
    b = number of brain regions (148), p = number of hyperparameters, e = number of epochs (different of each patient).
    The last dimension is of size e+1 since the last element is a graph learned using all the epochs combined
    :param id: ID of the patient corresponding to the W matrix
    :param frequency_band: Frequency band from which the W matrix was computed
    :param last_epoch: Whether to complete the data set with the graph learned from all epochs combined (True) or with
    all the graphs learned from single epochs (False)
    :return: data_set_alpha, data_set_beta, data_set_delta, data_set_gamma, data_set_theta
    Only the data set corresponding to "frequency_band" will be created/completed, the other will be returned unchanged
    """

    for i in range(params.shape[0]):
        data_set_temp = np.array([])
        if last_epoch:
            # if last epoch we only get the last element in the last dimension -> brain graph learned with all epochs
            # combined
            matrix = w[:, :, i, size_w[3] - 1]
            data_set_temp = create_set_epochs(data_set_temp, matrix, id)
        else:
            # get all the brain graphs learned using one epoch
            for j in range(size_w[3] - 1):
                matrix = w[:, :, i, j]
                data_set_temp = create_set_epochs(data_set_temp, matrix, id)
        # complete the data set corresponding to the same frequency band used to compute W
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


def create_all_sets(data_path, save_file=True, verbose=True, last_epoch=False,
                    saving_path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/'):
    """
    Creates training and test sets from a directory containing the brain graphs learned from MEG data. It creates one
    list per frequency band (alpha, beta, delta, gamma, theta) per regularization (l2 or log) per type (test/train) so
    20 data sets in total. Each data set is a list of size p (i.e. number of regularization hyperparameters) each element
    being a data set containing one sample per patient per epoch and D features corresponding to the upper triangular
    part of the adjacency matrix computed by the learning graph algorithm.

    :param data_path: string, path to the directory containing all the adjacency matrices learned by the graph learning
    algorithm
    :param save_file: bool, whether to save the created data sets (ech data set, i.e. each element of each list is
    saved as a separate file.). True by default.
    :param verbose: bool, whether to display information as the function runs. True by default.
    :param last_epoch: bool, whether to created datasets using only the graph learned using all epochs combined (True)
    or the graphs learned using all epochs separately (False). Default is false.
    :param saving_path: string, where to save the files, the directory must contain 1 subdirectory named "train" and one
    named "test".Default is the path we saved the file on the lab server.
    :return: 20 lists of 20 elements containing the train/test sets needed for creating our ML models
    """

    # get all the files in the directory data_path
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
            np.savetxt(saving_path + 'train/train_wl2_' + 'alpha_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_train_wl2_alpha[alpha], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wl2_' + 'beta_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_train_wl2_beta[alpha], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wl2_' + 'delta_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_train_wl2_delta[alpha], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wl2_' + 'gamma_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_train_wl2_gamma[alpha], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wl2_' + 'theta_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_train_wl2_theta[alpha], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wl2_' + 'alpha_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_test_wl2_alpha[alpha], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wl2_' + 'beta_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_test_wl2_beta[alpha], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wl2_' + 'delta_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_test_wl2_delta[alpha], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wl2_' + 'gamma_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_test_wl2_gamma[alpha], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wl2_' + 'theta_' + str(np.squeeze(alphas[alpha]))
                       + name + '.txt', final_data_set_test_wl2_theta[alpha], delimiter=' ')
            if verbose:
                print("all files for alpha = " + str(np.squeeze(alphas[alpha])) + " saved")

        for beta in range(betas.shape[0]):
            np.savetxt(saving_path + 'train/train_wlog_' + 'alpha_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_train_wlog_alpha[beta], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wlog_' + 'beta_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_train_wlog_beta[beta], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wlog_' + 'delta_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_train_wlog_delta[beta], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wlog_' + 'gamma_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_train_wlog_gamma[beta], delimiter=' ')
            np.savetxt(saving_path + 'train/train_wlog_' + 'theta_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_train_wlog_theta[beta], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wlog_' + 'alpha_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_test_wlog_alpha[beta], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wlog_' + 'beta_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_test_wlog_beta[beta], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wlog_' + 'delta_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_test_wlog_delta[beta], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wlog_' + 'gamma_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_test_wlog_gamma[beta], delimiter=' ')
            np.savetxt(saving_path + 'test/test_wlog_' + 'theta_' + str(np.squeeze(betas[beta]))
                       + name + '.txt', final_data_set_test_wlog_theta[beta], delimiter=' ')
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


def remove_col_lowvariance(datafrx, datafr_test, threshold):
    """
    It removes the columns with a lower value than the given threshold. This operation is done on the test and train set to be able to predict later.
    Args:
        datafr: a pandas DataFrame containing the train data
        datafr_test: a pandas DataFrame containing the test data
        threshold: a numerical value that sets the variance threshold below which the columns with lower variance will be removed  

    Returns: a DataFrame without the columns with variance below the threshold

    """
    variance = datafrx.var()
    
    dico = dict(list(enumerate(variance)))

    selected_col = [key for key, value in dico.items()
                    if value < threshold]
    datafrx_withoutvar = datafrx.drop(selected_col, axis=1)
    datafr_test_withoutvar = datafr_test.drop(selected_col, axis=1)

    return datafrx_withoutvar, datafr_test_withoutvar


def fix_outliers_median(datafr):
    """
    Remove the outliers by fixing its value to the median value of its column.
    Args:
        datafr: a pandas DataFrame containing the train data

    Returns: Dataframe with its outliers changed by the median value

    """
    median = datafr.median()
    std = datafr.std()
    value = datafr

    outlierspos = (value > median + 3 * std)
    outliersneg = (value < median - 3 * std)

    datafr_withoutvarmed = datafr.mask(outlierspos, other=median, axis=1)

    datafr_withoutvarmed = datafr.mask(outliersneg, other=median, axis=1)

    return datafr_withoutvarmed


def fix_outliers_std(datafr):
    """
    Remove the outliers by fixing its values to +/- 3 * standard deviation of its column
    Args:
        datafr: a pandas DataFrame containing the train data

    Returns: a pandas Dataframe with its outliers changed by +/- 3 * standard deviation of its column

    """

    median = datafr.median()
    std = datafr.std()
    value = datafr

    outlierspos = (value > median + 3 * std)
    outliersneg = (value < median - 3 * std)

    datafr_withoutvarstd = datafr.mask(outlierspos, other=3 * std, axis=1)

    datafr_withoutvarstd = datafr.mask(outliersneg, other=-3 * std, axis=1)

    return datafr_withoutvarstd


def load_data_set(band, regularization, type, parameter, epochs_combined=False,
                  path=r'/data_sets/'):
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
            - epochs_combined: bool, whether we want the data set from the brain graphs learned using all epochs
                               combined (True) or the ones using all epochs separately (False), False by default
            - path: string where the test and train files are

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


def predict_with_correlation(band, reg, param, epochs_combined, path):
    """
    Predicts the label of a test set using Pearson's coefficient of correlation. The label is assigned to a sample is
    the label of the train sample with the highest correlation with our test sample.

    :param band: string, "alpha", "beta", "delta" "gamma" or "theta". Frequency band of the test set we want to use
    to make prediction
    :param reg: string, "wl2" ou "wlog" regularization of the test set we want to use to make prediction
    :param param: float, hyperparameter value of the learning graph algorithm of the test set we want to use to make
    prediction, 20 possible value
    :param epochs_combined: bool, whether we want the test set to be the one derived from the graphs learned using all
    epochs combined (True) or not (False)
    :param path: string, where the test and train files are
    :return: y_pred: ndarray (N, ) the predicted labels, y_true: ndarray (N, ) the true label from our test set
    """
    # load the data sets
    test_set, y_test = load_data_set(band=band, regularization=reg, parameter=param,
                                     epochs_combined=epochs_combined, path=path, type="test")
    train_set, y_train = load_data_set(band=band, regularization=reg, parameter=param,
                                       epochs_combined=epochs_combined, path=path, type="train")
    y_pred = []
    # compute all the correlations to find the maximal one between each train and test sample
    for i in range(test_set.shape[0]):
        max_correlation = -1
        for j in range(train_set.shape[0]):
            correlation = np.corrcoef(test_set[i, :], train_set[j, :])[0, 1]
            if correlation >= max_correlation:
                max_correlation = correlation
                pred = y_train[j]
        y_pred.append(pred)

    return y_pred, y_test


def compute_benchmark(band, reg, params, epochs_combined=False, path=r'/data_sets/',
                      verbose=True):
    """
    Computes the accuracy of a specific band and regularization and with a prediction made with correlations
    instead of with an ML model.

    :param band: string, "alpha", "beta", "delta" "gamma" or "theta". Frequency band of the dataset used to make
    predictions and measure accuracy
    :param reg: string, "wl2" ou "wlog" regularization f the dataset used to make predictions and measure accuracy
    :param params: float, hyperparameter value of the learning graph algorithm of the dataset used to make
    predictions and measure accuracy
    :param epochs_combined: bool, whether we want the dataset to be the one derived from the graphs learned using all
    epochs combined (True) or not (False), default is False
    :param path: string, where the test and train files are
    :param verbose: bool, whether to display information as the function runs. True by default.
    :return: a dataframe with: regularization, band, hyperparameter and accuracy computed for each setting
    """

    # creates results data frame
    accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'accuracy'])
    # compute the prediction and the accuracy for each parameter in params and add it to the results table
    for i, param in enumerate(params):
        y_pred, y_true = predict_with_correlation(band, reg, param, epochs_combined, path)
        accuracy = compute_accuracy(y_pred, y_true)
        new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': param, 'accuracy': accuracy}, name=i)
        accuracy_table = accuracy_table.append(new_row, ignore_index=False)
        if verbose:
            print("accuracy for parameter: " + str(param) + " computed")

    return accuracy_table


def compute_accuracy(pred_ids, real_ids):
    """
    Computes the accuracy of a prediction wrt to the real labels

    :param pred_ids: predicted labels
    :param real_ids: true labels
    :return: accuracy
    """

    # takes two lists as arguments ordered in the same fashion to compare the ids stored in them
    boolean_matrix = (pred_ids == real_ids)
    accuracy = np.sum(boolean_matrix) / len(pred_ids)

    return accuracy


# function to create the two CSVs with sparsity values for all epochs
def save_sparsities_all_epochs(data_path, save_path):
    sparsity_all_epochs_alpha_l2, sparsity_all_epochs_alpha_log, sparsity_all_epochs_beta_l2, sparsity_all_epochs_beta_log, \
    sparsity_all_epochs_delta_l2, sparsity_all_epochs_delta_log, sparsity_all_epochs_gamma_l2, sparsity_all_epochs_gamma_log, \
    sparsity_all_epochs_theta_l2, sparsity_all_epochs_theta_log = get_sparsities_all_epochs(data_path)

    matrix_l2 = np.vstack((sparsity_all_epochs_alpha_l2,sparsity_all_epochs_beta_l2))
    matrix_l2 = np.vstack((matrix_l2, sparsity_all_epochs_delta_l2))
    matrix_l2 = np.vstack((matrix_l2, sparsity_all_epochs_gamma_l2))
    matrix_l2 = np.vstack((matrix_l2, sparsity_all_epochs_theta_l2))

    matrix_l2 = matrix_l2.T

    matrix_log = np.vstack((sparsity_all_epochs_alpha_log, sparsity_all_epochs_beta_log))
    matrix_log = np.vstack((matrix_log, sparsity_all_epochs_delta_log))
    matrix_log = np.vstack((matrix_log, sparsity_all_epochs_gamma_log))
    matrix_log = np.vstack((matrix_log, sparsity_all_epochs_theta_log))

    matrix_log = matrix_log.T

    l2_sparsities_df = pd.DataFrame(data = matrix_l2, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55','0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'], columns = ['alpha', 'beta', 'delta', 'gamma', 'theta'])
    log_sparsities_df = pd.DataFrame(data = matrix_log, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5', '0.55','0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'], columns= ['alpha', 'beta', 'delta', 'gamma', 'theta'])

    l2_sparsities_df.to_csv(save_path + 'sparsities_all_epochs_l2.csv')
    log_sparsities_df.to_csv(save_path + 'sparsities_all_epochs_log.csv')

    print("Done saving sparsity CSV files for all epochs")


# function to compute the sparsity of one matrix
def compute_sparsity(matrix):
    number_of_elements = matrix.shape[0]*matrix.shape[1]
    max_value = matrix.max()
    threshold = 0.05*max_value
    indices_almost_zero = np.where(matrix < threshold, 1, 0)
    total_number_of_almost_zeros = np.sum(indices_almost_zero)
    sparsity = total_number_of_almost_zeros / number_of_elements
    return sparsity


# function to compute the sparsities for all epochs
def get_sparsities_all_epochs(data_path):
    path = os.path.join(data_path, '**/*.mat')
    files = glob.glob(path, recursive=True)

    sparsity_all_epochs_alpha_l2 = []
    sparsity_all_epochs_alpha_log = []

    sparsity_all_epochs_beta_l2 = []
    sparsity_all_epochs_beta_log = []

    sparsity_all_epochs_delta_l2 = []
    sparsity_all_epochs_delta_log = []

    sparsity_all_epochs_gamma_l2 = []
    sparsity_all_epochs_gamma_log = []

    sparsity_all_epochs_theta_l2 = []
    sparsity_all_epochs_theta_log = []

    for file in files:
        sparsity_one_person_l2 = []
        sparsity_one_person_log = []
        session, alphas, betas, id, frequency_band, size_wl2, size_wlog, wl2, wlog = load_mat_file(file)

        if frequency_band == 'alpha':
            for i, alpha in enumerate(alphas):
                matrix = wl2[:, :, i, size_wl2[3]-1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_l2.append(sparsity)

            for i, beta in enumerate(betas):
                matrix = wlog[:, :, i, size_wlog[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_log.append(sparsity)

            if len(sparsity_all_epochs_alpha_l2) == 0:
                sparsity_all_epochs_alpha_l2 = sparsity_one_person_l2
                sparsity_all_epochs_alpha_l2 = np.asarray(sparsity_all_epochs_alpha_l2)
            else:
                sparsity_one_person_l2 = np.asarray(sparsity_one_person_l2)
                sparsity_all_epochs_alpha_l2 = np.vstack((sparsity_all_epochs_alpha_l2, sparsity_one_person_l2))

            if len(sparsity_all_epochs_alpha_log) == 0:
                sparsity_all_epochs_alpha_log = sparsity_one_person_log
                sparsity_all_epochs_alpha_log = np.asarray(sparsity_all_epochs_alpha_log)
            else:
                sparsity_one_person_log = np.asarray(sparsity_one_person_log)
                sparsity_all_epochs_alpha_log = np.vstack((sparsity_all_epochs_alpha_log, sparsity_one_person_log))

        if frequency_band == 'beta':
            for i, alpha in enumerate(alphas):
                matrix = wl2[:, :, i, size_wl2[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_l2.append(sparsity)

            for i, beta in enumerate(betas):
                matrix = wlog[:, :, i, size_wlog[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_log.append(sparsity)

            if len(sparsity_all_epochs_beta_l2) == 0:
                sparsity_all_epochs_beta_l2 = sparsity_one_person_l2
                sparsity_all_epochs_beta_l2 = np.asarray(sparsity_all_epochs_beta_l2)
            else:
                sparsity_one_person_l2 = np.asarray(sparsity_one_person_l2)
                sparsity_all_epochs_beta_l2 = np.vstack((sparsity_all_epochs_beta_l2, sparsity_one_person_l2))

            if len(sparsity_all_epochs_beta_log) == 0:
                sparsity_all_epochs_beta_log = sparsity_one_person_log
                sparsity_all_epochs_beta_log = np.asarray(sparsity_all_epochs_beta_log)
            else:
                sparsity_one_person_log = np.asarray(sparsity_one_person_log)
                sparsity_all_epochs_beta_log = np.vstack((sparsity_all_epochs_beta_log, sparsity_one_person_log))

        if frequency_band == 'delta':
            for i, alpha in enumerate(alphas):
                matrix = wl2[:, :, i, size_wl2[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_l2.append(sparsity)

            for i, beta in enumerate(betas):
                matrix = wlog[:, :, i, size_wlog[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_log.append(sparsity)

            if len(sparsity_all_epochs_delta_l2) == 0:
                sparsity_all_epochs_delta_l2 = sparsity_one_person_l2
                sparsity_all_epochs_delta_l2 = np.asarray(sparsity_all_epochs_delta_l2)
            else:
                sparsity_one_person_l2 = np.asarray(sparsity_one_person_l2)
                sparsity_all_epochs_delta_l2 = np.vstack((sparsity_all_epochs_delta_l2, sparsity_one_person_l2))

            if len(sparsity_all_epochs_delta_log) == 0:
                sparsity_all_epochs_delta_log = sparsity_one_person_log
                sparsity_all_epochs_delta_log = np.asarray(sparsity_all_epochs_delta_log)
            else:
                sparsity_one_person_log = np.asarray(sparsity_one_person_log)
                sparsity_all_epochs_delta_log = np.vstack((sparsity_all_epochs_delta_log, sparsity_one_person_log))

        if frequency_band == 'gamma':
            for i, alpha in enumerate(alphas):
                matrix = wl2[:, :, i, size_wl2[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_l2.append(sparsity)

            for i, beta in enumerate(betas):
                matrix = wlog[:, :, i, size_wlog[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_log.append(sparsity)

            if len(sparsity_all_epochs_gamma_l2) == 0:
                sparsity_all_epochs_gamma_l2 = sparsity_one_person_l2
                sparsity_all_epochs_gamma_l2 = np.asarray(sparsity_all_epochs_gamma_l2)
            else:
                sparsity_one_person_l2 = np.asarray(sparsity_one_person_l2)
                sparsity_all_epochs_gamma_l2 = np.vstack((sparsity_all_epochs_gamma_l2, sparsity_one_person_l2))

            if len(sparsity_all_epochs_gamma_log) == 0:
                sparsity_all_epochs_gamma_log = sparsity_one_person_log
                sparsity_all_epochs_gamma_log = np.asarray(sparsity_all_epochs_gamma_log)
            else:
                sparsity_one_person_log = np.asarray(sparsity_one_person_log)
                sparsity_all_epochs_gamma_log = np.vstack((sparsity_all_epochs_gamma_log, sparsity_one_person_log))

        if frequency_band == 'theta':
            for i, alpha in enumerate(alphas):
                matrix = wl2[:, :, i, size_wl2[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_l2.append(sparsity)

            for i, beta in enumerate(betas):
                matrix = wlog[:, :, i, size_wlog[3] - 1]
                sparsity = compute_sparsity(matrix)
                sparsity_one_person_log.append(sparsity)

            if len(sparsity_all_epochs_theta_l2) == 0:
                sparsity_all_epochs_theta_l2 = sparsity_one_person_l2
                sparsity_all_epochs_theta_l2 = np.asarray(sparsity_all_epochs_theta_l2)
            else:
                sparsity_one_person_l2 = np.asarray(sparsity_one_person_l2)
                sparsity_all_epochs_theta_l2 = np.vstack((sparsity_all_epochs_theta_l2, sparsity_one_person_l2))

            if len(sparsity_all_epochs_theta_log) == 0:
                sparsity_all_epochs_theta_log = sparsity_one_person_log
                sparsity_all_epochs_theta_log = np.asarray(sparsity_all_epochs_theta_log)
            else:
                sparsity_one_person_log = np.asarray(sparsity_one_person_log)
                sparsity_all_epochs_theta_log = np.vstack((sparsity_all_epochs_theta_log, sparsity_one_person_log))

    sparsity_all_epochs_alpha_l2 = np.mean(sparsity_all_epochs_alpha_l2, axis=0)
    sparsity_all_epochs_alpha_log = np.mean(sparsity_all_epochs_alpha_log, axis=0)

    sparsity_all_epochs_beta_l2 = np.mean(sparsity_all_epochs_beta_l2, axis=0)
    sparsity_all_epochs_beta_log = np.mean(sparsity_all_epochs_beta_log, axis=0)

    sparsity_all_epochs_delta_l2 = np.mean(sparsity_all_epochs_delta_l2, axis=0)
    sparsity_all_epochs_delta_log = np.mean(sparsity_all_epochs_delta_log, axis=0)

    sparsity_all_epochs_gamma_l2 = np.mean(sparsity_all_epochs_gamma_l2, axis=0)
    sparsity_all_epochs_gamma_log = np.mean(sparsity_all_epochs_gamma_log, axis=0)

    sparsity_all_epochs_theta_l2 = np.mean(sparsity_all_epochs_theta_l2, axis=0)
    sparsity_all_epochs_theta_log = np.mean(sparsity_all_epochs_theta_log, axis=0)

    return sparsity_all_epochs_alpha_l2,sparsity_all_epochs_alpha_log,sparsity_all_epochs_beta_l2,sparsity_all_epochs_beta_log,\
           sparsity_all_epochs_delta_l2, sparsity_all_epochs_delta_log, sparsity_all_epochs_gamma_l2, sparsity_all_epochs_gamma_log,\
           sparsity_all_epochs_theta_l2, sparsity_all_epochs_theta_log


# function to get all sparsities per epoch per band per regularization
def compute_sparsities_per_epoch(data_path):
    path = os.path.join(data_path, '**/*.mat')
    files = glob.glob(path, recursive=True)

    # third dimension should be 84*2 on final data but 3 on the test
    matrix_alpha_l2 = np.zeros((20, 17, 168))
    matrix_alpha_log = np.zeros((20, 17, 168))
    matrix_beta_l2 = np.zeros((20, 17, 168))
    matrix_beta_log = np.zeros((20, 17, 168))
    matrix_delta_l2 = np.zeros((20, 17, 168))
    matrix_delta_log = np.zeros((20, 17, 168))
    matrix_gamma_l2 = np.zeros((20, 17, 168))
    matrix_gamma_log = np.zeros((20, 17, 168))
    matrix_theta_l2 = np.zeros((20, 17, 168))
    matrix_theta_log = np.zeros((20, 17, 168))

    count_alpha = 0
    count_beta = 0
    count_delta = 0
    count_gamma = 0
    count_theta = 0

    for file in files:
        epoch_matrix_alpha_l2 = []
        epoch_matrix_alpha_log = []
        epoch_matrix_beta_l2 = []
        epoch_matrix_beta_log = []
        epoch_matrix_delta_l2 = []
        epoch_matrix_delta_log = []
        epoch_matrix_gamma_l2 = []
        epoch_matrix_gamma_log = []
        epoch_matrix_theta_l2 = []
        epoch_matrix_theta_log = []


        session, alphas, betas, id, frequency_band, size_wl2, size_wlog, wl2, wlog = load_mat_file(file)

        if frequency_band == 'alpha':
            for alpha in range(len(alphas)):
                line_of_epochs_l2 = []
                # here the range is not fixed to size_wl2[3]-1 because not all people have the same
                # number of epochs --> the min value is 17 so we will stop there
                for epoch in range(17):
                    matrix = wl2[:, :, alpha, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_l2.append(sparsity)

                if len(epoch_matrix_alpha_l2) == 0:
                    epoch_matrix_alpha_l2 = line_of_epochs_l2
                    epoch_matrix_alpha_l2 = np.asarray(epoch_matrix_alpha_l2)
                else:
                    line_of_epochs_l2 = np.asarray(line_of_epochs_l2)
                    epoch_matrix_alpha_l2 = np.vstack((epoch_matrix_alpha_l2, line_of_epochs_l2))

            matrix_alpha_l2[:, :, count_alpha] = epoch_matrix_alpha_l2

            for i, beta in enumerate(betas):
                line_of_epochs_log = []
                for epoch in range(17):
                    matrix = wlog[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_log.append(sparsity)
                if len(epoch_matrix_alpha_log) == 0:
                    epoch_matrix_alpha_log = line_of_epochs_log
                    epoch_matrix_alpha_log = np.asarray(epoch_matrix_alpha_log)
                else:
                    line_of_epochs_log = np.asarray(line_of_epochs_log)
                    epoch_matrix_alpha_log = np.vstack((epoch_matrix_alpha_log, line_of_epochs_log))

            matrix_alpha_log[:, :, count_alpha] = epoch_matrix_alpha_log
            count_alpha += 1

        elif frequency_band == 'beta':
            for i, alpha in enumerate(alphas):
                line_of_epochs_l2 = []
                # here the range is not fixed to size_wl2[3]-1 because not all people have the same
                # number of epochs --> the min value is 17 so we will stop there
                for epoch in range(17):
                    matrix = wl2[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_l2.append(sparsity)
                if len(epoch_matrix_beta_l2) == 0:
                    epoch_matrix_beta_l2 = line_of_epochs_l2
                    epoch_matrix_beta_l2 = np.asarray(epoch_matrix_beta_l2)
                else:
                    line_of_epochs_l2 = np.asarray(line_of_epochs_l2)
                    epoch_matrix_beta_l2 = np.vstack((epoch_matrix_beta_l2, line_of_epochs_l2))
            matrix_beta_l2[:, :, count_beta] = epoch_matrix_beta_l2

            for i, beta in enumerate(betas):
                line_of_epochs_log = []
                for epoch in range(17):
                    matrix = wlog[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_log.append(sparsity)
                if len(epoch_matrix_beta_log) == 0:
                    epoch_matrix_beta_log = line_of_epochs_log
                    epoch_matrix_beta_log = np.asarray(epoch_matrix_beta_log)
                else:
                    line_of_epochs_log = np.asarray(line_of_epochs_log)
                    epoch_matrix_beta_log = np.vstack((epoch_matrix_beta_log, line_of_epochs_log))
            matrix_beta_log[:, :, count_beta] = epoch_matrix_beta_log
            count_beta += 1

        elif frequency_band == 'delta':
            for i, alpha in enumerate(alphas):
                line_of_epochs_l2 = []
                # here the range is not fixed to size_wl2[3]-1 because not all people have the same
                # number of epochs --> the min value is 17 so we will stop there
                for epoch in range(17):
                    matrix = wl2[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_l2.append(sparsity)
                if len(epoch_matrix_delta_l2) == 0:
                    epoch_matrix_delta_l2 = line_of_epochs_l2
                    epoch_matrix_delta_l2 = np.asarray(epoch_matrix_delta_l2)
                else:
                    line_of_epochs_l2 = np.asarray(line_of_epochs_l2)
                    epoch_matrix_delta_l2 = np.vstack((epoch_matrix_delta_l2, line_of_epochs_l2))
            matrix_delta_l2[:, :, count_delta] = epoch_matrix_delta_l2

            for i, beta in enumerate(betas):
                line_of_epochs_log = []
                for epoch in range(17):
                    matrix = wlog[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_log.append(sparsity)
                if len(epoch_matrix_delta_log) == 0:
                    epoch_matrix_delta_log = line_of_epochs_log
                    epoch_matrix_delta_log = np.asarray(epoch_matrix_delta_log)
                else:
                    line_of_epochs_log = np.asarray(line_of_epochs_log)
                    epoch_matrix_delta_log = np.vstack((epoch_matrix_delta_log, line_of_epochs_log))
            matrix_delta_log[:, :, count_delta] = epoch_matrix_delta_log
            count_delta += 1

        elif frequency_band == 'gamma':
            for i, alpha in enumerate(alphas):
                line_of_epochs_l2 = []
                # here the range is not fixed to size_wl2[3]-1 because not all people have the same
                # number of epochs --> the min value is 17 so we will stop there
                for epoch in range(17):
                    matrix = wl2[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_l2.append(sparsity)
                if len(epoch_matrix_gamma_l2) == 0:
                    epoch_matrix_gamma_l2 = line_of_epochs_l2
                    epoch_matrix_gamma_l2 = np.asarray(epoch_matrix_gamma_l2)
                else:
                    line_of_epochs_l2 = np.asarray(line_of_epochs_l2)
                    epoch_matrix_gamma_l2 = np.vstack((epoch_matrix_gamma_l2, line_of_epochs_l2))
            matrix_gamma_l2[:, :, count_gamma] = epoch_matrix_gamma_l2
            for i, beta in enumerate(betas):
                line_of_epochs_log = []
                for epoch in range(17):
                    matrix = wlog[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_log.append(sparsity)
                if len(epoch_matrix_gamma_log) == 0:
                    epoch_matrix_gamma_log = line_of_epochs_log
                    epoch_matrix_gamma_log = np.asarray(epoch_matrix_gamma_log)
                else:
                    line_of_epochs_log = np.asarray(line_of_epochs_log)
                    epoch_matrix_gamma_log = np.vstack((epoch_matrix_gamma_log, line_of_epochs_log))
            matrix_gamma_log[:, :, count_gamma] = epoch_matrix_gamma_log
            count_gamma += 1

        elif frequency_band == 'theta':
            for i, alpha in enumerate(alphas):
                line_of_epochs_l2 = []
                # here the range is not fixed to size_wl2[3]-1 because not all people have the same
                # number of epochs --> the min value is 17 so we will stop there
                for epoch in range(17):
                    matrix = wl2[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_l2.append(sparsity)
                if len(epoch_matrix_theta_l2) == 0:
                    epoch_matrix_theta_l2 = line_of_epochs_l2
                    epoch_matrix_theta_l2 = np.asarray(epoch_matrix_theta_l2)
                else:
                    line_of_epochs_l2 = np.asarray(line_of_epochs_l2)
                    epoch_matrix_theta_l2 = np.vstack((epoch_matrix_theta_l2, line_of_epochs_l2))
            matrix_theta_l2[:, :, count_theta] = epoch_matrix_theta_l2
            for i, beta in enumerate(betas):
                line_of_epochs_log = []
                for epoch in range(17):
                    matrix = wlog[:, :, i, epoch]
                    sparsity = compute_sparsity(matrix)
                    line_of_epochs_log.append(sparsity)
                if len(epoch_matrix_theta_log) == 0:
                    epoch_matrix_theta_log = line_of_epochs_log
                    epoch_matrix_theta_log = np.asarray(epoch_matrix_theta_log)
                else:
                    line_of_epochs_log = np.asarray(line_of_epochs_log)
                    epoch_matrix_theta_log = np.vstack((epoch_matrix_theta_log, line_of_epochs_log))
            matrix_theta_log[:, :, count_theta] = epoch_matrix_theta_log
            count_theta += 1

    matrix_alpha_l2 = np.mean(matrix_alpha_l2, axis=2)
    matrix_alpha_log = np.mean(matrix_alpha_log, axis=2)
    matrix_beta_l2 = np.mean(matrix_beta_l2, axis=2)
    matrix_beta_log = np.mean(matrix_beta_log, axis=2)
    matrix_delta_l2 = np.mean(matrix_delta_l2, axis=2)
    matrix_delta_log = np.mean(matrix_delta_log, axis=2)
    matrix_gamma_l2 = np.mean(matrix_gamma_l2, axis=2)
    matrix_gamma_log = np.mean(matrix_gamma_log, axis=2)
    matrix_theta_l2 = np.mean(matrix_theta_l2, axis=2)
    matrix_theta_log = np.mean(matrix_theta_log, axis=2)

    return matrix_alpha_l2, matrix_alpha_log, matrix_beta_l2, matrix_beta_log, matrix_delta_l2, matrix_delta_log,\
           matrix_gamma_l2, matrix_gamma_log, matrix_theta_l2, matrix_theta_log


# function that creates all (10) CSV files for each band and regularization with the sparsity per epoch and parameter
def save_sparsities_each_epochs(data_path, save_path):
    matrix_alpha_l2, matrix_alpha_log, matrix_beta_l2, matrix_beta_log, matrix_delta_l2, matrix_delta_log, matrix_gamma_l2, \
    matrix_gamma_log, matrix_theta_l2, matrix_theta_log = compute_sparsities_per_epoch(data_path)

    alpha_band_l2_sparsities_df = pd.DataFrame(data=matrix_alpha_l2, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])
    alpha_band_log_sparsities_df = pd.DataFrame(data=matrix_alpha_log, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                                columns=[np.linspace(1,17,17)])
    beta_band_l2_sparsities_df = pd.DataFrame(data=matrix_beta_l2, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                              columns=[np.linspace(1,17,17)])
    beta_band_log_sparsities_df = pd.DataFrame(data=matrix_beta_log, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])
    delta_band_l2_sparsities_df = pd.DataFrame(data=matrix_delta_l2, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])
    delta_band_log_sparsities_df = pd.DataFrame(data=matrix_delta_log, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                                columns=[np.linspace(1,17,17)])
    gamma_band_l2_sparsities_df = pd.DataFrame(data=matrix_gamma_l2, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])
    gamma_band_log_sparsities_df = pd.DataFrame(data=matrix_gamma_log, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])
    theta_band_l2_sparsities_df = pd.DataFrame(data=matrix_theta_l2, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])
    theta_band_log_sparsities_df = pd.DataFrame(data=matrix_theta_log, index=['0.05', '0.1', '0.15', '0.20', '0.25', '0.3', '0.35', '0.4', '0.45',
                                                        '0.5', '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9', '0.95', '1'],
                                               columns=[np.linspace(1,17,17)])

    alpha_band_l2_sparsities_df.to_csv(save_path + 'sparsities_alpha_band_l2.csv')
    alpha_band_log_sparsities_df.to_csv(save_path + 'sparsities_alpha_band_log.csv')

    beta_band_l2_sparsities_df.to_csv(save_path + 'sparsities_beta_band_l2.csv')
    beta_band_log_sparsities_df.to_csv(save_path + 'sparsities_beta_band_log.csv')

    delta_band_l2_sparsities_df.to_csv(save_path + 'sparsities_delta_band_l2.csv')
    delta_band_log_sparsities_df.to_csv(save_path + 'sparsities_delta_band_log.csv')

    gamma_band_l2_sparsities_df.to_csv(save_path + 'sparsities_gamma_band_l2.csv')
    gamma_band_log_sparsities_df.to_csv(save_path + 'sparsities_gamma_band_log.csv')

    theta_band_l2_sparsities_df.to_csv(save_path + 'sparsities_theta_band_l2.csv')
    theta_band_log_sparsities_df.to_csv(save_path + 'sparsities_theta_band_log.csv')

    print("Done saving sparsity CSV files per epochs")
