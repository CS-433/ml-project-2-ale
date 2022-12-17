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
    plt.close("all")
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

'''
def plot_accuracies_wrt_parameter_per_band(path_tables, save_path, save_fig=False):
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
                plt.show()'''


def find_labels(filename):
    if "alpha" in filename:
        return "alpha"
    elif "beta" in filename:
        return "beta"
    elif "delta" in filename:
        return "delta"
    elif "gamma" in filename:
        return "gamma"
    elif "theta" in filename:
        return "theta"


def plot_accuracies_wrt_parameter(path_tables, save_path, model_name, regularization, save_fig=False,
                                  y_scale=np.linspace(0, 1, 21, endpoint=True), all_epochs=False):
    path = path_tables
    all_files = Path(path).glob('*.csv')
    params = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499999999999999, 0.7, 0.75, 0.8,
              0.85, 0.9, 0.95, 1.]
    # files = glob.glob(path, recursive=true)
    for file in all_files:
        df = pd.read_csv(file)
        x = df["alpha/beta"]
        y = df["accuracy"]
        name_of_file = os.path.basename(os.path.normpath(file))
        name_of_file = name_of_file[:-4]
        condition = "all_epochs" in name_of_file
        title = model_name + "_" + regularization + "_accuracies"
        if not condition and not all_epochs:
            if regularization in name_of_file:
                plt.plot(x, y, label=find_labels(name_of_file))
                plt.legend()
                plt.title(title)
                plt.legend()
                plt.ylabel('accuracy')
                plt.xlabel('learning graph parameter')
                plt.xticks(params)
                plt.xticks(rotation=90)
                plt.yticks(y_scale)
                plt.grid(True)
                plt.tight_layout()
        elif condition and all_epochs:
            if regularization in name_of_file:
                plt.plot(x, y, label=find_labels(name_of_file))
                plt.legend()
                plt.title(title)
                plt.legend()
                plt.ylabel('accuracy')
                plt.xlabel('learning graph parameter')
                plt.xticks(params)
                plt.xticks(rotation=90)
                plt.yticks(y_scale)
                plt.grid(True)
                plt.tight_layout()
    if save_fig:
        plt.savefig(save_path + title + ".png")
    plt.show()


def best_accuracies_df_all_epochs(params, dataframe):
    new_accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'accuracy'])
    bands = ["alpha", "beta", "delta", "gamma", "theta"]
    for band in bands:
        for i, param in enumerate(params):
            sub_df = dataframe.loc[dataframe["alpha/beta"] == param]
            sub_df = sub_df.loc[sub_df["band"] == band]
            idx_best_accuracy = sub_df["accuracy"].idxmax()
            reg = sub_df['reg'][idx_best_accuracy]
            accuracy = sub_df['accuracy'][idx_best_accuracy]
            new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': param, 'accuracy': accuracy}, name=i)
            new_accuracy_table = new_accuracy_table.append(new_row, ignore_index=False)
    return new_accuracy_table


def plot_accuracies_wrt_parameter_from1file(path_tables, save_path, model_name, regularization, save_fig=False,
                                             y_scale=np.linspace(0, 1, 21, endpoint=True)):
    path = path_tables
    all_files = Path(path).glob('*.csv')
    params = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499999999999999, 0.7, 0.75, 0.8,
              0.85, 0.9, 0.95, 1.]

    for file in all_files:
        df = pd.read_csv(file)
        name_of_file = os.path.basename(os.path.normpath(file))
        name_of_file = name_of_file[:-4]
        title = model_name + "_" + regularization + "_all_epochs_accuracies"

        if "all_epochs" in name_of_file:
            if regularization in name_of_file:
                best_accuracies_df = best_accuracies_df_all_epochs(params=params, dataframe=df)

                alpha_accuracy = best_accuracies_df.loc[best_accuracies_df["band"] == "alpha"]
                alpha_accuracy = alpha_accuracy['accuracy']
                alpha_accuracy = alpha_accuracy.to_numpy()
                beta_accuracy = best_accuracies_df.loc[best_accuracies_df["band"] == "beta"]
                beta_accuracy = beta_accuracy['accuracy']
                beta_accuracy = beta_accuracy.to_numpy()
                delta_accuracy = best_accuracies_df.loc[best_accuracies_df["band"] == "delta"]
                delta_accuracy = delta_accuracy['accuracy']
                delta_accuracy = delta_accuracy.to_numpy()
                gamma_accuracy = best_accuracies_df.loc[best_accuracies_df["band"] == "gamma"]
                gamma_accuracy = gamma_accuracy['accuracy']
                gamma_accuracy = gamma_accuracy.to_numpy()
                theta_accuracy = best_accuracies_df.loc[best_accuracies_df["band"] == "theta"]
                theta_accuracy = theta_accuracy['accuracy']
                theta_accuracy = theta_accuracy.to_numpy()
                plt.plot(params, alpha_accuracy, label='alpha')
                plt.plot(params, beta_accuracy, label='beta')
                plt.plot(params, delta_accuracy, label='delta')
                plt.plot(params, gamma_accuracy, label='gamma')
                plt.plot(params, theta_accuracy, label='theta')
                plt.legend()

                plt.title(title)
                plt.xlabel('learning graph parameter')
                plt.ylabel('accuracy')
                plt.xticks(params)
                plt.xticks(rotation=90)
                plt.yticks(y_scale)
                plt.grid(True)
                plt.tight_layout()
                if save_fig:
                    plt.savefig(save_path + title + ".png")
                plt.show()
