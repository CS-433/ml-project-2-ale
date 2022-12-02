import pandas as pd

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
# alphas = np.linspace(0.05, 1, num=20, endpoint=True)
alphas = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45,0.5, 0.55, 0.6,
          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
'''alphas[2] = 0.15000000000000002
alphas[12] = 0.6499999999999999
alphas[6] = 0.35000000000000003'''
print(alphas)

reg = "wl2"
band = "alpha"

"""accuracies = np.zeros((len(bands), len(alphas)))
best_Cs = np.zeros((len(bands), len(alphas)))
best_gammas = np.zeros((len(bands), len(alphas)))
best_kernels = np.zeros((len(bands), len(alphas)))"""
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy'])

for j, alpha in enumerate(alphas):
    # x_train, y_train = load_data_set(band, reg, "train", alpha, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
    # x_test, y_test = load_data_set(band, reg, "test", alpha, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
    x_train, y_train = load_data_set(band, reg, "train", alpha, path=r'../data_sets/')
    x_test, y_test = load_data_set(band, reg, "test", alpha, path=r'../data_sets/')
    title = "confusion_matrix_" + reg + "_" + band + "_" + str(alpha)
    accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_fig=False, title=title)
    new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': alpha, 'C': C, 'gamma': gamma, 'kernel': kernel,
                              'accuracy': accuracy}, name=j)
    accuracy_table = accuracy_table.append(new_row, ignore_index=False)

print(accuracy_table)
accuracy_table.to_csv(path_or_buf='test_accuracy_table.csv')
