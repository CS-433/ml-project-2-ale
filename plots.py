import numpy as np
from models import *
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


# TODO: MAKE it work
def plot_similarity_matrix(prediction_scores, title="similarity matrix", save_fig=False, show_values=False):
    # x = np.arange(1, 49)
    # y = np.arange(1, 49)
    x = np.arange(1, 4)
    y = np.arange(1, 4)

    fig, ax = plt.subplots()
    im = ax.imshow(prediction_scores)
    # Setting the labels
    ax.set_xticks(x)
    ax.set_yticks(y)
    # labeling respective list entries
    # ax.set_xticklabels(str(x))
    # ax.set_yticklabels(str(y))
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    ax.set_title(title)
    im = ax.imshow(prediction_scores)
    if show_values:
        for i in range(len(y)):
            for j in range(len(x)):
                text = plt.text(j, i, round(prediction_scores[i, j],4),
                               ha="center", va="center", color="w")

    fig.tight_layout()
    if save_fig:
        plt.savefig(title + ".png")
    plt.show()


def plot_confusion_matrix(model, x_test, y_test, ids, save_fig=False, title="confusion_matrix"):
    ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=ids, xticks_rotation="horizontal")
    plt.tight_layout()
    if save_fig:
        plt.savefig(title + ".png")
    plt.show()
