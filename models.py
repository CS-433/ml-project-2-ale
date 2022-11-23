import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# need to visualize the data -> if sparse no normalizing but scaling
# check outliers
# maybe normalize instead of standardize
def SVM_predict(x_train, y_train, x_test, C=1, gamma='scale', kernel='rbf'):
    '''
        Pre-process the data, fit an SVM model on the training set and predicts the labels associated to the test set
        :param x_train:
        :param y_train:
        :param x_test:
        :return:
    '''

    # TODO: check if other pre-processing is needed
    # TODO: maybe other param to tune
    # define the model (standardization and then SVM)
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=C, gamma=gamma))
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    prediction_scores = model.decision_function(x_test)

    return model, y_pred_test, train_accuracy, prediction_scores
