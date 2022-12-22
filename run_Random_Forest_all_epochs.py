from models import *
from helpers import *

sparsity_parameters = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
                        0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

# can be set to either "wl2" or "wlog depending on which the type of regularization used to generate our data sets
# we want to explore
reg = "wl2"

# compute predictions for each frequency bands
bands = ["alpha", "beta", "delta", "gamma", "theta"]
# creates the accuracy table in which our results will be saved
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'n estimators', 'max depth', 'min samples split',
                                       'min samples leaf', 'accuracy'])
# parameters that seem to work well for some settings using all epochs separately
max_depths = [50, 100]
all_n_estimators = [500, 1000]

print("starting predictions for RF all epochs and regularization: " + reg)
for band in bands:
    print("starting predictions for: " + band)
    for j, spar in enumerate(sparsity_parameters):
        for max_depth in max_depths:
            for n_est in all_n_estimators:
                x_train, y_train = load_data_set(band, reg, "train", spar, path=r'data_sets/Train_set/',
                                                 epochs_combined=True)
                x_test, y_test = load_data_set(band, reg, "test", spar, path=r'data_sets/Test_set/',
                                               epochs_combined=True)
                title = "confusion_matrix_RF_" + reg + "_" + band + "_" + str(spar) + '_all_epochs_combined'
                accuracy, n_estimators, max_depth, \
                min_samples_split, min_samples_leaf = RandomForest_tune_predict_evaluate(x_train, y_train, x_test, y_test,
                                                                                save_fig=False, grid_search=False,
                                                                                n_jobs=20, title=title,
                                                                                save_path=r'plots/',
                                                                                default_max_depth=max_depth,
                                                                                default_n_estimators=n_est)
                new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': spar, 'n estimators': n_estimators,
                                          'max depth': max_depth, 'min samples split': min_samples_split,
                                          'min samples leaf': min_samples_leaf, 'accuracy': accuracy}, name=j)
                accuracy_table = accuracy_table.append(new_row, ignore_index=False)
        print("predictions done with band: " + band + " and alpha: " + str(spar))

print("all predictions done")

accuracy_table.to_csv(path_or_buf=r'../results/RandomForest/RF_accuracy_table_all_epochs.csv')
print("accuracy table successfully saved")
