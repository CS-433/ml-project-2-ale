import pandas as pd
from models import *
from helpers import *

# train the SVM model on the all epochs train sets without doing a grid search but using defing hyper-parameters
# and compute the accuracy on the test set

# we explore the train/test sets for all sparsity parameters
sparsity_parameters = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
                       0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

# Here we define the training and test set that are used
# regularization can be either "wlog" or "wl2"
reg = "wlog"
# the band is either "alpha", "beta", "delta", "gamma" or "theta"
band = "gamma"
# creating the accuracy table used to store results
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy'])

print("starting predictions for: " + band)
for j, spar in enumerate(sparsity_parameters):
    x_train, y_train = load_data_set(band, reg, "train", spar, path=r'data_sets/Train_set/',
                                     epochs_combined=True)
    x_test, y_test = load_data_set(band, reg, "test", spar, path=r'data_sets/Test_set/',
                                   epochs_combined=True)
    title = "confusion_matrix_" + reg + "_" + band + "_" + str(spar) +'_all_epochs_combined'
    accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test,
                                                            save_path=r'plots/',
                                                            C_params=[0.1, 1, 10, 100],
                                                            gamma_params=[10, 1, 0.1, 0.01, 0.001, 'scale'],
                                                            kernel_params=['rbf', 'sigmoid'],
                                                            save_fig=False, title="confusion_matrix",
                                                            grid_search=False, default_C=1,
                                                            default_gamma='scale', default_kernel='rbf')
    new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': spar, 'C': C, 'gamma': gamma, 'kernel': kernel,
                        'accuracy': accuracy, 'nb_features:': x_train.size()}, name=j)
    accuracy_table = accuracy_table.append(new_row, ignore_index=False)
    print("prediction done with parameter: " + str(spar))

print("all predictions done")

name_of_file = 'SVM_accuracy_table_all_epochs' + reg + '_' + band + '.csv'
accuracy_table.to_csv(path_or_buf=r'sample_data/' + name_of_file)
print("accuracy table successfully saved")
