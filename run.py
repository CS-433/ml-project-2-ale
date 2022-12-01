from models import *
from plots import *
from helpers import *
from sklearn.model_selection import GridSearchCV

'''
##### load the training sets #####

x_train, y_train = load_csv_data(r'Hamid_ML4Science_ALE/train_beta0.1')
x_test, y_test = load_csv_data(r'Hamid_ML4Science_ALE/test_beta0.1')

##### grid search #####
params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=params,
    cv=5,
    n_jobs=5,
    verbose=1
)

grid_search.fit(x_train, y_train)
# {'C': 0.1, 'gamma': 10, 'kernel': 'rbf'}
print(grid_search.best_params_)
best_C, best_gamma, best_kernel = grid_search.best_params_

#### prediction #####
model, pred, train_accuracy = SVM_predict(x_train, y_train, x_test, C=best_C, gamma=best_gamma, kernel=best_kernel)

#### evaluation of the model ####
test_accuracy = accuracy_score(y_test, pred)
# plot similarity matrix
# plot_similarity_matrix(pred_score, save_fig=True, show_values=True)
plot_confusion_matrix(model, x_test, y_test, ids=y_test) '''


bands = ["alpha", "beta", "delta", "gamma", "theta"]
# TODO find the alphas
alphas = []
reg = "wl2"
accuracies = np.zeros((len(bands), len(alphas)))
best_Cs = np.zeros((len(bands), len(alphas)))
best_gammas = np.zeros((len(bands), len(alphas)))
best_kernels = np.zeros((len(bands), len(alphas)))

for i, band in enumerate(bands):
    for j, alpha in enumerate(alphas):
        x_train, y_train = load_data_set(band, reg, "train", alpha, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
        x_test, y_test = load_data_set(band, reg, "test", alpha, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
        title = "confusion_matrix_" + reg + "_" + band + "_" + str(alpha)
        accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_fig=False, title=title)
        accuracies[i, j] = accuracy
        best_Cs[i, j] = C
        best_gammas[i, j] = gamma
        best_kernels[i, j] = kernel

# TODO: get min accuracies and which param
# TODO: maybe save the accuracies in a file
print(accuracies)
