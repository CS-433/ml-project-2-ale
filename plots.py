import numpy as np
from models import *
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(y_test, y_pred, accuracy, save_path, save_fig=False, title="confusion_matrix"):
    """
        Plots the confusion matrix using the true and the predicted labels
        Arguments:
            - y_test: numpy array of size (D), labels corresponding to the test set, true labels
            - y_pred: numpy array of size (D), predicted labels
            - accuracy: accuracy of the model used to generate the y_pred
            - save_path: where the figure will be saved if save_fig=True
            - save_fig: whether the figure will be saved, default = False
            - title: title of the plot and name of the file if the figure is saved
        """
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, xticks_rotation="vertical", cmap="hot", colorbar=False,
                                            normalize=None, include_values=False)
    plt.title(title + " , accuracy: " + str(round(accuracy, 5)))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path + title + ".png")
    else:
        plt.show()
