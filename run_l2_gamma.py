import pandas as pd
from models import *
from helpers import *

alphas = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45,0.5, 0.55, 0.6,
          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]

reg = "wl2"
band = "gamma"
accuracy_table = pd.DataFrame(columns=['reg', 'band', 'alpha/beta', 'C', 'gamma', 'kernel', 'accuracy'])

for j, alpha in enumerate(alphas):
    x_train, y_train = load_data_set(band, reg, "train", alpha, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
    x_test, y_test = load_data_set(band, reg, "test", alpha, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
    title = "confusion_matrix_" + reg + "_" + band + "_" + str(alpha)
    accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_fig=True, title=title, save_path=r'../../Data3/Hamid_ML4Science_ALE/plots/')
    new_row = pd.Series(data={'reg': reg, 'band': band, 'alpha/beta': alpha, 'C': C, 'gamma': gamma, 'kernel': kernel,
                              'accuracy': accuracy}, name=j)
    accuracy_table = accuracy_table.append(new_row, ignore_index=False)

accuracy_table.to_csv(path_or_buf=r'../../Data3/Hamid_ML4Science_ALE/accuracy_table_l2_alpha.csv')
