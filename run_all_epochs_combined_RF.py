import pandas as pd
from models import *
from helpers import *

alphas = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45,0.5, 0.55, 0.6,
          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

reg = "wl2"
bands = ["alpha", "beta", "delta", "gamma", "theta"]
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'n estimators', 'max depth', 'min samples split',
                                       'min samples leaf', 'accuracy'])
# parameters that seem to work well for some settings using all epochs separately
max_depths = [50, 100]
all_n_estimators = [500, 1000]


print("starting predictions for regularization: " + reg)
for band in bands:
    print("starting predictions for: " + band)
    for j, alpha in enumerate(alphas):
        for max_depth in max_depths:
            for n_est in all_n_estimators:
                x_train, y_train = load_data_set(band, reg, "train", alpha, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/',
                                                 epochs_combined=True)
                x_test, y_test = load_data_set(band, reg, "test", alpha, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/',
                                               epochs_combined=True)
                title = "confusion_matrix_RF_" + reg + "_" + band + "_" + str(alpha) + '_all_epochs_combined'
                accuracy, n_estimators, max_depth, \
                min_samples_split, min_samples_leaf = RandomForest_tune_predict_evaluate(x_train, y_train, x_test, y_test,
                                                                                save_fig=False, grid_search=False,
                                                                                n_jobs=20, title=title,
                                                                                save_path=r'../../../Data3/Hamid_ML4Science_ALE/RandomForest/plots/',
                                                                                default_max_depth=max_depth,
                                                                                default_n_estimators=n_est)
                new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': alpha, 'n estimators': n_estimators,
                                          'max depth': max_depth, 'min samples split': min_samples_split,
                                          'min samples leaf': min_samples_leaf, 'accuracy': accuracy}, name=j)
                accuracy_table = accuracy_table.append(new_row, ignore_index=False)
        print("predictions done with band: " + band + " and alpha: " + str(alpha))

print("all predictions done")

accuracy_table.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/RandomForest/accuracy_table_RF_l2_all_epochs.csv')
print("accuracy table successfully saved")
