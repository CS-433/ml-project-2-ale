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


def find_band_labels(filename):
    """
    Finds which frequency band is in the filename and returns it as a string

    :param filename: string, name of the file
    :return: "alpha", "beta", "delta", "gamma" or "theta" depending on which string is present in filename
    """
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


def find_reg_labels(filename):
    """
    Finds which regularization and which model was used to generate by looking at the string present "filename"

    :param filename: string, name of the file
    :return: string, the name of the model and which regularization was used
    """
    if "correlations" in filename:
        if "l2" in filename:
            return "correlation l2"
        elif "log" in filename:
            return "correlation log"
    elif "RandomForest" in filename:
        if "l2" in filename:
            return "RandomForest l2"
        elif "log" in filename:
            return "RandomForest log"
    elif "SVM" in filename:
        if "l2" in filename:
            return "SVM l2"
        elif "log" in filename:
            return "SVM log"


def find_line_style(filename):
    """
    Returns a matplot lib type of line depending on what string are present in filename (i.e. with which model were
    these results generated)

    :param filename: name of file
    :return: string, "dashed" if correlations is in filename, "solid" otherwise
    """
    if "correlations" in filename:
        return "dashed"
    else:
        return "solid"


def find_line_color(filename):
    """
    Returns different colors depending on the string present in filename, to be able to plot bands and regularization
    in different colors.

    :param filename: string, name of file
    :return: string, a color name different depending of the band and the regularization present in filename
    """
    if "alpha" in filename:
        if "l2" in filename:
            return "blue"
        elif "log" in filename:
            return "dodgerblue"
    elif "beta" in filename:
        if "l2" in filename:
            return "#ff7f0e"
        elif "log" in filename:
            return "darkgoldenrod"
    elif "delta" in filename:
        if "l2" in filename:
            return "darkgreen"
        elif "log" in filename:
            return "limegreen"
    elif "gamma" in filename:
        if "l2" in filename:
            return "darkred"
        elif "log" in filename:
            return "#d62728"
    elif "theta" in filename:
        if "l2" in filename:
            return "indigo"
        elif "log" in filename:
            return "mediumorchid"


def plot_accuracies_wrt_parameter(path_tables, save_path, model_name, regularization, params, save_fig=False,
                                  y_scale=np.linspace(0, 1, 21, endpoint=True), all_epochs=False):
    """
    Plot the accuracies (with regards to the parameters params) of all the files contained in filename. It will generate
    one plot with all the bands either from files containing results from single epochs data sets or all epochs combined
    data sets and for one type of regularization

    :param path_tables: string, directory where the datasets are stored (as .csv)
    :param save_path: string, directory where the plots will be saved if save_fig = True
    :param model_name: string, name of the model used to generate the result files, is used in the name of the file when
    saved
    :param regularization: string, type of regularization used to generate the results we want to plot
    :param params: list of double, the values corresponding to the x-axis
    :param save_fig: bool, whether to save the figure, default = False
    :param y_scale: list of float, the y-scale to be used in the plot, default = 0 to 1 with 0.05 increments
    :param all_epochs: bool, whether to plot the results generated from all epochs combined dataset or single epochs
    generated data sets, default = False
    """
    path = path_tables
    all_files = Path(path).glob('*.csv')

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
                plt.plot(x, y, label=find_band_labels(name_of_file))
        elif condition and all_epochs:
            if regularization in name_of_file:
                plt.plot(x, y, label=find_band_labels(name_of_file))
    plt.legend(fontsize=12)
    plt.ylabel('accuracy', fontsize=25)
    plt.xlabel('learning graph parameter', fontsize=25)
    plt.xticks(params, rotation=90, fontsize=15)
    plt.yticks(y_scale, fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    if save_fig:
        plt.savefig(save_path + title + ".png")
    plt.show()


def create_df_best_accuracies_all_epochs(params, dataframe):
    """
    Creates a data frame with only the best accuracy per parameter in params when we have a file with
    more than one accuracy for each parameter in params

    :param params: list of float, i.e. list of parameters for which accuracies were computed
    :param dataframe: panda data frame with multiple accuracies for each parameter in param
    :return: panda data frame with only the best accuracy per parameter retained
    """
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


def plot_accuracies_wrt_parameter_from1file(path_tables, save_path, model_name, regularization, params, save_fig=False,
                                             y_scale=np.linspace(0, 1, 21, endpoint=True)):
    """
    Plot the accuracies (with regards to the parameters params) of the file contained in filename. It will generate
    one plot with all the bands either from one file containing results from single epochs data sets or all epochs combined
    data sets and for one type of regularization. In this case, all the accuracies for all the bands are stored in the
    same file and multiple accuracies are stored for each parameter in params. The function first creates a new data
    frame containig only one accuracy per parameter (the highest one) and then plots all the data from the .csv file in
    one plot. Such .csv file are only generated when we use results generated with all epochs combined data sets.

    :param path_tables: string, directory where the datasets are stored (as .csv)
    :param save_path: string, directory where the plots will be saved if save_fig = True
    :param model_name: string, name of the model used to generate the result files, is used in the name of the file when
    saved
    :param regularization: string, type of regularization used to generate the results we want to plot
    :param params: list of double, the values corresponding to the x-axis
    :param save_fig: bool, whether to save the figure, default = False
    :param y_scale: list of float, the y-scale to be used in the plot, default = 0 to 1 with 0.05 increments
    """
    path = path_tables
    all_files = Path(path).glob('*.csv')

    for file in all_files:
        df = pd.read_csv(file)
        name_of_file = os.path.basename(os.path.normpath(file))
        name_of_file = name_of_file[:-4]
        title = model_name + "_" + regularization + "_all_epochs_accuracies"

        if "all_epochs" in name_of_file:
            if regularization in name_of_file:
                best_accuracies_df = create_df_best_accuracies_all_epochs(params=params, dataframe=df)
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
                plt.legend(fontsize=12)
                plt.ylabel('accuracy', fontsize=25)
                plt.xlabel('learning graph parameter', fontsize=25)
                plt.xticks(params, rotation=90, fontsize=15)
                plt.yticks(y_scale, fontsize=15)
                plt.grid(True)
                plt.tight_layout()
                if save_fig:
                    plt.savefig(save_path + title + ".png")
                plt.show()


def plot_accuracies_all_settings_1band(path_tables, save_path, band, params, save_fig=False,
                                       y_scale=np.linspace(0, 1, 21, endpoint=True)):
    """
    Plot the accuracies from different models on the y-axis and params on the x-axis for a single frequency band.
    All the accuracies data are stored in .csv files in the directory given by path_tables. The correlation models are
    plotted with dashed lines and the ML models with solid lines.

    :param path_tables: string, directory where the datasets are stored (as .csv)
    :param save_path: string, directory where the plots will be saved if save_fig = True
    :param band: string, the frequency band used to generate the result files we want to plot
    :param model_name: string, name of the model used to generate the result files, is used in the name of the file when
    saved
    :param regularization: string, type of regularization used to generate the results we want to plot
    :param params: list of double, the values corresponding to the x-axis
    :param save_fig: bool, whether to save the figure, default = False
    :param y_scale: list of float, the y-scale to be used in the plot, default = 0 to 1 with 0.05 increments
    """
    path = os.path.join(path_tables, '**/*.csv')
    all_files = glob.glob(path, recursive=True)

    for file in all_files:
        name_of_file = os.path.basename(os.path.normpath(file))
        name_of_file = name_of_file[:-4]
        condition = "all_epochs" in name_of_file
        title = band + "_all_settings_accuracies"
        if band in name_of_file and not condition:
            df = pd.read_csv(file)
            x = df["alpha/beta"]
            y = df["accuracy"]
            label = find_reg_labels(name_of_file)
            plt.plot(x, y, label=label, linestyle=find_line_style(name_of_file), color=find_line_color(name_of_file))
            plt.legend(fontsize=12)
            plt.ylabel('accuracy', fontsize=25)
            plt.xlabel('learning graph parameter', fontsize=25)
            plt.xticks(params, rotation=90, fontsize=15)
            plt.yticks(y_scale, fontsize=15)
            plt.grid(True)
            plt.tight_layout()

    if save_fig:
        plt.savefig(save_path + title + ".png")
    plt.show()


# function to make the two plot with all epochs combined
def plot_sparsities_all_epochs(csv_files_path, save_path, save_fig=False):
    path = csv_files_path
    all_files = Path(path).glob('*.csv')

    for file in all_files:
        df = pd.read_csv(file)
        name_of_file = os.path.basename(os.path.normpath(file))
        title = name_of_file[:-4]
        if "all_epochs" in title:
            if 'l2' in title:
                alphas = df.iloc[:, 0]
                sparsity_alpha = df['alpha']
                sparsity_beta = df['beta']
                sparsity_delta = df['delta']
                sparsity_gamma = df['gamma']
                sparsity_theta = df['theta']

                plt.plot(alphas, sparsity_alpha, label='alpha band')
                plt.plot(alphas, sparsity_beta, label='beta band')
                plt.plot(alphas, sparsity_delta, label='delta band')
                plt.plot(alphas, sparsity_gamma, label='gamma band')
                plt.plot(alphas, sparsity_theta, label='theta band')
                plt.legend()

                plt.title(title + ": plot")
                plt.xlabel('alpha')
                plt.ylabel('sparsity')
                plt.xticks(alphas)
                plt.xticks(rotation=90)
                plt.grid(True)
                plt.tight_layout()

                if save_fig:
                    plt.savefig(save_path + title + ".png")
                else:
                    plt.show()

                plt.clf()

            elif 'log' in title:
                betas  = df.iloc[:, 0]
                sparsity_alpha = df['alpha']
                sparsity_beta = df['beta']
                sparsity_delta = df['delta']
                sparsity_gamma = df['gamma']
                sparsity_theta = df['theta']

                plt.plot(betas, sparsity_alpha, label='alpha band')
                plt.plot(betas, sparsity_beta, label='beta band')
                plt.plot(betas, sparsity_delta, label='delta band')
                plt.plot(betas, sparsity_gamma, label='gamma band')
                plt.plot(betas, sparsity_theta, label='theta band')
                plt.legend()

                plt.title(title + ": plot")
                plt.xlabel('beta')
                plt.ylabel('sparsity')
                plt.xticks(betas)
                plt.xticks(rotation=90)
                plt.grid(True)
                plt.tight_layout()

                if save_fig:
                    plt.savefig(save_path + title + ".png")
                else:
                    plt.show()

    print("Done saving sparsity plots for all epochs")


def plot_sparsities_per_epoch(csv_files_path, save_path, save_fig=False):
    # function to make the 10 plots for each band with all 17 epochs each as a line
    path = csv_files_path
    all_files = Path(path).glob('*.csv')

    for file in all_files:
        df = pd.read_csv(file)
        name_of_file = os.path.basename(os.path.normpath(file))
        title = name_of_file[:-4]
        if 'l2' in title:
            alphas = df.iloc[:, 0]
            for epoch in np.linspace(1,17,17):
                plt.plot(alphas, df.iloc[:, int(epoch)], linewidth=0.1)

            plt.title(title + ": plot")
            plt.xlabel('alpha')
            plt.ylabel('sparsity')
            plt.xticks(alphas)
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.tight_layout()

            if save_fig:
                plt.savefig(save_path + title + ".png")
            else:
                plt.show()

            plt.clf()

        elif 'log' in title:
            betas = df.iloc[:, 0]
            for epoch in np.linspace(1, 17, 17):
                plt.plot(betas, df.iloc[:, int(epoch)], linewidth=0.1)

            plt.title(title + ": plot")
            plt.xlabel('beta')
            plt.ylabel('sparsity')
            plt.xticks(betas)
            plt.xticks(rotation=90)
            plt.grid(True)
            plt.tight_layout()

            if save_fig:
                plt.savefig(save_path + title + ".png")
            else:
                plt.show()
        plt.clf()

    print("Done saving sparsity plots per epochs")