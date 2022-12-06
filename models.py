import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from plots import *
from helpers import *
from sklearn.model_selection import GridSearchCV
import pandas as pd

# need to visualize the data -> if sparse no normalizing but scaling
# check outliers
# maybe normalize instead of standardize


def SVM_predict(x_train, y_train, x_test, C=1, gamma='scale', kernel='rbf'):
    """
        Pre-process the data, fit an SVM model on the training set and predicts the labels associated to the test set
        Arguments:
            - x_train: numpy array of size (N_train,D), training set
            - y_train: numpy array of size (D), labels corresponding to the training set
            - x_test: numpy array of size (N_test, D), test set
            - C: C hyperparameter used in the SVM model, by default C = 1
            - gamma: gamma hyperparameter used in the SVM model, by default gamma='scale', i.e. gamma = 1/(D * X.var())
            - kernel: kernel hyperparameter used in the SVM model, by default kernel = rbf
        Returns:
            - y_pred_test: the labels predicted by the SVM model for our test set
            - train_accuracy: the training accuracy of our model

    """
    # TODO: check if other pre-processing is needed
    # TODO: maybe other param to tune
    # define the model (standardization and then SVM) and fit it
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))
    model.fit(x_train, y_train)
    # make predictions
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    # computation of accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)

    return y_pred_test, train_accuracy


def SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_path='', C_params=[0.1, 1, 10, 100],
                              gamma_params=[10, 1, 0.1, 0.01, 0.001, 'scale'], kernel_params=['rbf', 'sigmoid'],
                              save_fig=False, title="confusion_matrix", grid_search=True):
    """
       Performs a grid search + 5-folds CV over all the given parameter to find the best SVM model for our training set
       Fits this best model to our training set and use it to predict the labels of our test set
       Evaluates the accuracy of the model and plots the resulting confusion matrix
       Args:
           - x_train: numpy array of size (N_train,D), training set
           - y_train: numpy array of size (D), labels corresponding to the training set
           - x_test: numpy array of size (N_test, D), test set
           - y_test: numpy array of size (D), labels corresponding to the test set
           - save_path: path to which the confusion matrix will be saved, default = directory where the script is run
           - C_params: list of C hyperparameters used in the grid search of the SVM model
                        by default C_params = [0.1, 1, 10, 100]
           - gamma_params: list of gamma hyperparameters used in the grid search of the SVM model
                            by default gamma_params = [10, 1, 0.1, 0.01, 0.001, 0.0001]
           - kernel_params: list of kernel hyperparameters used in the grid search of the SVM model
                            by default kernel_params = ['rbf', 'poly', 'sigmoid']
           - save_fig: whether to save the generated figure with the confusion matrix, False by default
           - title: title of the generated figure and name as which it will be saved if save_fig=True
                    by default title = "confusion_matrix"
       Returns:
           - test_accuracy: the test accuracy of the best SVM model
           - best_C, best_gamma, best_kernel: the SVM hyperparameters used in our SVM model (selected with 5-fold CV)
       """

    if grid_search:
        # grid search
        params = {'C': C_params, 'gamma': gamma_params, 'kernel': kernel_params}
        grid_search = GridSearchCV(estimator=SVC(), param_grid=params, cv=5, n_jobs=5, verbose=1)
        grid_search.fit(x_train, y_train)
        best_C = grid_search.best_params_["C"]
        best_gamma = grid_search.best_params_["gamma"]
        best_kernel = grid_search.best_params_["kernel"]
        results = pd.DataFrame(grid_search.cv_results_)
        # results.to_csv(path_or_buf=r'../../Data3/Hamid_ML4Science_ALE/accuracy_table_grid_search_test.csv')
        # results.to_csv(path_or_buf=r'accuracy_table_grid_search_test.csv')
        # print(results[['param_kernel', 'param_gamma', 'param_C', 'mean_test_score']])
    else:
        best_C = 1
        best_gamma = 'scale'
        best_kernel = 'rbf'

    # prediction
    pred, train_accuracy = SVM_predict(x_train, y_train, x_test, C=best_C, gamma=best_gamma, kernel=best_kernel)

    # evaluation of the model
    test_accuracy = accuracy_score(y_test, pred)

    plot_confusion_matrix(y_test, pred, accuracy=test_accuracy, save_fig=save_fig, title=title, save_path=save_path)

    return test_accuracy, best_C, best_gamma, best_kernel
