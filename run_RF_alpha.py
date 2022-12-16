from models import *
from helpers import *

alphas = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

reg = "wlog"
band = "alpha"
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'n estimators', 'max depth', 'min samples split',
                                       'min samples leaf', 'accuracy'])

print("starting predictions")

for j, alpha in enumerate(alphas):
    x_train, y_train = load_data_set(band, reg, "train", alpha, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/')
    x_test, y_test = load_data_set(band, reg, "test", alpha, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/')
    title = "confusion_matrix_RF" + reg + "_" + band + "_" + str(alpha)
    accuracy, n_estimators, max_depth, \
        min_samples_split, min_samples_leaf = RandomForest_tune_predict_evaluate(x_train, y_train, x_test,
                                                                                 y_test, save_fig=True,
                                                                                 grid_search=True, n_jobs=20,
                                                                                 title=title, save_path=r'../../../Data3/Hamid_ML4Science_ALE/RandomForest/plots/')
    new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': alpha, 'n estimators': n_estimators,
                              'max depth': max_depth, 'min samples split': min_samples_split,
                              'min samples leaf': min_samples_leaf, 'accuracy': accuracy}, name=j)
    accuracy_table = accuracy_table.append(new_row, ignore_index=False)
    print("prediction done with alpha: " + str(alpha))

print("all predictions done")

accuracy_table.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/RandomForest/accuracy_table_log_alpha.csv')
print("accuracy table successfully saved")
