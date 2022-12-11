import numpy as np
from models import *
import matplotlib.pyplot as plt
import pandas as pd
import os
from pathlib import Path
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


def plot_accuracies_wrt_parameter(path_tables, save_path, save_fig=False):
    path = path_tables
    all_files = Path(path).glob('*.csv')
    # files = glob.glob(path, recursive=true)

    for file in all_files:
        df = pd.read_csv(file)
        name_of_file = os.path.basename(os.path.normpath(file))
        title = name_of_file[:-4]


        if "all_epochs" in title:
            if "l2" in title:
                alphas = df["alpha/beta"]
                alphas = alphas.truncate(after=19)
                alphas = alphas.to_numpy()
                alpha_accuracy = df.loc[df["band"] == "alpha"]
                alpha_accuracy = alpha_accuracy['accuracy']
                alpha_accuracy = alpha_accuracy.to_numpy()
                beta_accuracy = df.loc[df["band"] == "beta"]
                beta_accuracy = beta_accuracy['accuracy']
                beta_accuracy = beta_accuracy.to_numpy()
                delta_accuracy = df.loc[df["band"] == "delta"]
                delta_accuracy = delta_accuracy['accuracy']
                delta_accuracy = delta_accuracy.to_numpy()
                gamma_accuracy = df.loc[df["band"] == "gamma"]
                gamma_accuracy = gamma_accuracy['accuracy']
                gamma_accuracy = gamma_accuracy.to_numpy()
                theta_accuracy = df.loc[df["band"] == "theta"]
                theta_accuracy = theta_accuracy['accuracy']
                theta_accuracy = theta_accuracy.to_numpy()

                plt.plot(alphas, alpha_accuracy, label='alpha')
                plt.plot(alphas, beta_accuracy, label='beta')
                plt.plot(alphas, delta_accuracy, label='delta')
                plt.plot(alphas, gamma_accuracy, label='gamma')
                plt.plot(alphas, theta_accuracy, label='theta')
                plt.legend()

                plt.title(title + ": plot")
                plt.xlabel('alpha')
                plt.ylabel('accuracy')
                if save_fig:
                    plt.savefig(save_path + title + ".png")
                else:
                    plt.show()

            if "log" in title:
                alphas = df["alpha/beta"]
                alphas = alphas.truncate(after=19)
                alphas = alphas.to_numpy()
                alpha_accuracy = df.loc[df["band"] == "alpha"]
                alpha_accuracy = alpha_accuracy['accuracy']
                alpha_accuracy = alpha_accuracy.to_numpy()
                beta_accuracy = df.loc[df["band"] == "beta"]
                beta_accuracy = beta_accuracy['accuracy']
                beta_accuracy = beta_accuracy.to_numpy()
                delta_accuracy = df.loc[df["band"] == "delta"]
                delta_accuracy = delta_accuracy['accuracy']
                delta_accuracy = delta_accuracy.to_numpy()
                gamma_accuracy = df.loc[df["band"] == "gamma"]
                gamma_accuracy = gamma_accuracy['accuracy']
                gamma_accuracy = gamma_accuracy.to_numpy()
                theta_accuracy = df.loc[df["band"] == "theta"]
                theta_accuracy = theta_accuracy['accuracy']
                theta_accuracy = theta_accuracy.to_numpy()

                plt.plot(alphas, alpha_accuracy, label='alpha')
                plt.plot(alphas, beta_accuracy, label='beta')
                plt.plot(alphas, delta_accuracy, label='delta')
                plt.plot(alphas, gamma_accuracy, label='gamma')
                plt.plot(alphas, theta_accuracy, label='theta')
                plt.legend()

                plt.title(title + ": plot")
                plt.xlabel('beta')
                plt.ylabel('accuracy')
                if save_fig:
                    plt.savefig(save_path + title + ".png")
                else:
                    plt.show()

        else:
            df.plot(x='alpha/beta', y='accuracy', kind='line')
            plt.title(title + ": plot")
            plt.ylabel('accuracy')
            if "l2" in title:
                plt.xlabel('alpha')
            elif "log" in title:
                plt.xlabel('beta')

            if save_fig:
                plt.savefig(save_path + title + ".png")
            else:
                plt.show()
