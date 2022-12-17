from models import *
from sklearn.model_selection import cross_validate
from sklearn.svm import SVC


def validation(train_setx,train_sety, test_setx,threshold, best_C, best_kernel, best_gamma):

    best_threshh=0.0
    accuracyy=0.0
    _scoring = ['accuracy']
    for ind, T in enumerate(threshold):
        x_train, x_test = remove_col_lowvariance(pd.DataFrame(train_setx), pd.DataFrame(test_setx), T)

        result = cross_validate(estimator=SVC(C=best_C, gamma=best_gamma, kernel= best_kernel),
                             X=x_train,
                             y=train_sety,
                             cv=5,
                             scoring=_scoring,
                             return_train_score=True,
                             n_jobs=20)

        if result['test_accuracy'].mean()>accuracyy:
            best_threshh = T
            accuracyy=result['test_accuracy'].mean()
        print('Done for threshold: ', threshold)
    return best_threshh, accuracyy


betas = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
        0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]


threshold_l2= np.linspace(0.00005,0.01,35, endpoint=True)

bands = ["alpha", "beta", "delta", "gamma", "theta"]



reg = "wl2"


accuracy_table_al = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy_test', 'accuracy_validation', 'nb_features'])
accuracy_table_be = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy_test', 'accuracy_validation', 'nb_features'])
accuracy_table_de = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy_test', 'accuracy_validation', 'nb_features'])
accuracy_table_ga = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy_test', 'accuracy_validation', 'nb_features'])
accuracy_table_th = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy_test', 'accuracy_validation', 'nb_features'])


print("starting predictions")
for band in bands:
    print("starting predictions for: " + band)
    for j, beta in enumerate(betas):


        x_train, y_train = load_data_set(band, reg, "train", beta, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/',
                                                 epochs_combined=False)
        x_test, y_test = load_data_set(band, reg, "test", beta, path=r'../../../Data3/Hamid_ML4Science_ALE/data_sets/',
                                               epochs_combined=False)
        print("Data loaded")
        best_thresh, accuracy_valid = validation(x_train,y_train, x_test,threshold_l2, 10, 'sigmoid', 'scale')
        print("Validation step done")
        x_train, x_test = remove_col_lowvariance(pd.DataFrame(x_train), pd.DataFrame(x_test), best_thresh)
        accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test,
                                                                                   save_path=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/plots/',
                                                                                   C_params=[0.1, 1, 10, 100],
                                                                                   gamma_params=[10, 1, 0.1, 0.01, 0.001, 'scale'],
                                                                                   kernel_params=['rbf', 'sigmoid'],
                                                                                   save_fig=False, title="confusion_matrix",
                                                                                   grid_search=False, default_C=10,
                                                                                   default_gamma='scale', default_kernel='sigmoid')

        new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': beta, 'C': C, 'gamma': gamma, 'kernel': kernel,
                                                     'accuracy_test': accuracy, 'accuracy_validation':accuracy_valid,'nb_features': len(pd.DataFrame(x_train).columns)}, name=j)
        print("test accuracy done")
        if band == 'alpha':
            accuracy_table_al = accuracy_table_al.append(new_row, ignore_index=False)
        if band == 'beta':
            accuracy_table_be = accuracy_table_be.append(new_row, ignore_index=False)
        if band == 'delta':
            accuracy_table_de = accuracy_table_de.append(new_row, ignore_index=False)
        if band == 'gamma':
            accuracy_table_ga = accuracy_table_ga.append(new_row, ignore_index=False)
        if band == 'theta':
            accuracy_table_th = accuracy_table_th.append(new_row, ignore_index=False)

        print("prediction done with beta: " + str(beta))

print("all predictions done")

accuracy_table_al.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/Final_results/accuracywithoutVar_table_l2_all_epochs_alpha.csv')
accuracy_table_be.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/Final_results/accuracywithoutVar_table_l2_all_epochs_beta.csv')
accuracy_table_de.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/Final_results/accuracywithoutVar_table_l2_all_epochs_delta.csv')
accuracy_table_ga.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/Final_results/accuracywithoutVar_table_l2_all_epochs_gamma.csv')
accuracy_table_th.to_csv(path_or_buf=r'../../../Data3/Hamid_ML4Science_ALE/SVMwithoutVar/Final_results/accuracywithoutVar_table_l2_all_epochs_theta.csv')

print("accuracy table successfully saved")
