import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from plots import *
from helpers import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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
                              save_fig=False, title="confusion_matrix", grid_search=True, default_C=1,
                              default_gamma='scale', default_kernel='rbf'):
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

    else:
        best_C = default_C
        best_gamma = default_gamma
        best_kernel = default_kernel

    # prediction
    pred, train_accuracy = SVM_predict(x_train, y_train, x_test, C=best_C, gamma=best_gamma, kernel=best_kernel)

    # evaluation of the model
    test_accuracy = accuracy_score(y_test, pred)

    # plot the results if save_fig = True
    plot_confusion_matrix(y_test, pred, accuracy=test_accuracy, save_fig=save_fig, title=title, save_path=save_path)

    return test_accuracy, best_C, best_gamma, best_kernel


def RandomForest_predict(x_train, y_train, x_test, n_estimators=100, max_depth=None,
                         min_samples_split=2, min_samples_leaf=1,
                         n_jobs=5):
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=n_estimators,
                                                                   max_depth=max_depth, min_samples_split=min_samples_split,
                                                                   min_samples_leaf=min_samples_leaf,
                                                                   n_jobs=n_jobs, random_state=1))
    model.fit(x_train, y_train)
    # make predictions
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    # computation of accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)

    return y_pred_test, train_accuracy


def RandomForest_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_path='', n_estimators_params=[100, 500, 1000],
                                       max_depth_params=[50, 100, 1000, None],
                                       min_samples_split_params=[2, 5, 10], min_samples_leaf_params=[1, 2, 4],
                                       n_jobs=5, save_fig=False,
                                       title="confusion_matrix", grid_search=True, default_n_estimators=100,
                                       default_max_depth=None, default_min_samples_split=2,
                                       default_min_samples_leaf=1):

    # TODO: is there more parameters to include in the grid search ???
    if grid_search:
        # grid search
        params = {'n_estimators': n_estimators_params, 'max_depth': max_depth_params,
                  'min_samples_split': min_samples_split_params, 'min_samples_leaf': min_samples_leaf_params}
        grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=1), param_grid=params, cv=3, n_jobs=20, verbose=1)
        grid_search.fit(x_train, y_train)
        best_n_estimators = grid_search.best_params_["n_estimators"]
        best_max_depth = grid_search.best_params_["max_depth"]
        best_min_samples_split = grid_search.best_params_["min_samples_split"]
        best_min_samples_leaf = grid_search.best_params_["min_samples_leaf"]
        print("grid search done")

    else:
        best_n_estimators = default_n_estimators
        best_max_depth = default_max_depth
        best_min_samples_split = default_min_samples_split
        best_min_samples_leaf = default_min_samples_leaf

    # prediction
    pred, train_accuracy = RandomForest_predict(x_train, y_train, x_test, n_estimators=best_n_estimators,
                                                max_depth=best_max_depth,
                                                min_samples_split=best_min_samples_split, min_samples_leaf=best_min_samples_leaf,
                                                n_jobs=n_jobs)
    print("predictions done")
    # evaluation of the model
    test_accuracy = accuracy_score(y_test, pred)

    # plot the results if save_fig = True
    plot_confusion_matrix(y_test, pred, accuracy=test_accuracy, save_fig=save_fig, title=title, save_path=save_path)

    return test_accuracy, best_n_estimators, best_max_depth, best_min_samples_split, best_min_samples_leaf


def validation_SVM(train_setx,train_sety, test_setx,threshold, best_C, best_kernel, best_gamma):
    """
    Cross validation to tune the SVM model. It iterates on a number of threshold to remove the features with a variance below it 
     Args:
        train_setx: an array returned by the function "load_data_set" which provides the matrix of data for the train set
        train_sety: an array returned by the function "load_data_set" which provides the matrix of data for the test set
        test_setx: an array returned by the function "load_data_set" which provides the IDs according to the train set
        threshold: either a linspace or a numerical value. The variance of the columns belows it will be removed
        best_C: a numerical value. This is a parameter of SVM. The best ones found by grid search are 1 or 10 
        best_kernel: a string. This is a parameter of SVM. The best ones found by grid search are "rbf" or "sigmoid" 
        best_gamma:  a string. This is a parameter of SVM. The best one found by grid search is "scale"

    Returns: return the best threshold which gives the higher accuracy, and its accuracy on the cross validation set 
"""

    best_threshh=0.0
    accuracyy=0.0
    _scoring = ['accuracy']
    #remove a number of columns according to its variance and compute the accuracy 
    for ind, T in enumerate(threshold):
        x_train, x_test = remove_col_lowvariance(pd.DataFrame(train_setx), pd.DataFrame(test_setx), T)

        result = cross_validate(estimator=SVC(C=best_C, gamma=best_gamma, kernel= best_kernel),
                             X=x_train,
                             y=train_sety,
                             cv=5,
                             scoring=_scoring,
                             return_train_score=True,
                             n_jobs=20)

        if result['test_accuracy'].mean()>accuracyy:
            best_threshh = T
            accuracyy=result['test_accuracy'].mean()
        print('Done for threshold: ', threshold)
    return best_threshh, accuracyy
