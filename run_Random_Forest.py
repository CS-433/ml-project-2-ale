from models import *
from helpers import *

#the parameters used to control the sparsity of our model
sparsity_parameters = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

#determining the regularisation used
reg = "wlog"

#the frequency bands used to load the data. A selection of them can be chosen as: ["alpha", "beta"]
band = "beta"

accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'n estimators', 'max depth', 'min samples split',
                                       'min samples leaf', 'accuracy'])

print("starting predictions")

for j, spar in enumerate(sparsity_parameters):
    x_train, y_train = load_data_set(band, reg, "train", spar, path=r'data_sets/Train_set/')
    x_test, y_test = load_data_set(band, reg, "test", spar, path=r'data_sets/Test_set/')
    title = "confusion_matrix_RF" + reg + "_" + band + "_" + str(spar)

    #compute the accuracy using Random Forest model
    accuracy, n_estimators, max_depth, \
        min_samples_split, min_samples_leaf = RandomForest_tune_predict_evaluate(x_train, y_train, x_test,
                                                                                 y_test, save_fig=True,
                                                                                 grid_search=True, n_jobs=20,
                                                                                 title=title, save_path=r'plots/')
    new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': spar, 'n estimators': n_estimators,
                              'max depth': max_depth, 'min samples split': min_samples_split,
                              'min samples leaf': min_samples_leaf, 'accuracy': accuracy}, name=j)
    accuracy_table = accuracy_table.append(new_row, ignore_index=False)
    print("prediction done with alpha: " + str(spar))

print("all predictions done")

#be careful to change the name of the file according to the regularisation 
accuracy_table.to_csv(path_or_buf=r'Sample_data/RF_accuracy_table_log.csv')
print("accuracy table successfully saved")
