import pandas as pd
from models import *
from helpers import *

#betas = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
#          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
betas = [0.6499999999999999]

threshold=np.linspace(5*10**-6,1*10**-3, 20, endpoint=True)
reg = "wlog"
#bands = ["alpha", "beta", "delta", "gamma", "theta"]
bands = ["beta"]
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy'])

print("starting predictions")

for band in bands:
    print("starting predictions for: " + band)
    for j, beta in enumerate(betas):
        for indice, value in enumerate(threshold):
            x_train, y_train = load_data_set(band, reg, "train", beta, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/',
                                             epochs_combined=True)
            x_test, y_test = load_data_set(band, reg, "test", beta, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/',
                                           epochs_combined=True)
            x_train, x_test = remove_col_lowvariance(pd.DataFrame(x_train), pd.DataFrame(x_test), value)
            title = "confusion_matrix_" + reg + "_" + band + "_" + str(beta) + +"_" + str(value)+'_all_epochs_combined'
            accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test,
                                                                   save_path=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/plots/',
                                                                   C_params=[0.1, 1, 10, 100],
                                                                   gamma_params=[10, 1, 0.1, 0.01, 0.001, 'scale'],
                                                                   kernel_params=['rbf', 'sigmoid'],
                                                                   save_fig=False, title="confusion_matrix",
                                                                   grid_search=False, default_C=1,
                                                                   default_gamma='scale', default_kernel='sigmoid')
            new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': beta, 'C': C, 'gamma': gamma, 'kernel': kernel,
                                      'accuracy': accuracy}, name=j)
            accuracy_table = accuracy_table.append(new_row, ignore_index=False)
            print("prediction done with threshold: " + str(value))
        print("prediction done with beta: " + str(beta))

print("all predictions done")

accuracy_table.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/accuracywithoutVar_table_log_all_epochs_betaband.csv')
print("accuracy table successfully saved")
