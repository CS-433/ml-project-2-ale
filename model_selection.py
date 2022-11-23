from models import *
from helpers import *
from sklearn.model_selection import GridSearchCV

# TODO: modify with the real data path
x_train, y_train = load_csv_data(r'Hamid_ML4Science_ALE/train_beta0.1')
x_train = np.append(x_train, x_train, axis=0)
y_train = np.append(y_train, y_train, axis=0)

# grid search to find the best parameters of SVM
params = {
    'C': [0.1, 1, 10, 100],
    'gamma': [10, 1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(
    estimator=SVC(),
    param_grid=params,
    # TODO: change cv to 5 with real data
    cv=2,
    n_jobs=5,
    verbose=1
)

grid_search.fit(x_train, y_train)
# {'C': 0.1, 'gamma': 10, 'kernel': 'rbf'}
print(grid_search.best_params_)
