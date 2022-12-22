from helpers import *

bands = ["alpha", "beta", "delta", "gamma", "theta"]
reg = 'wlog'
params = [0.05, 0.1, 0.15000000000000002, 0.2, 0.25, 0.3, 0.35000000000000003, 0.4, 0.45, 0.5, 0.55, 0.6,
          0.6499999999999999, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
print("starting predictions for regularization: " + reg)
for band in bands:
    file_name = "accuracy_table_correlations_all_epochs_" + reg + "_" + band + ".csv"
    save_path = r'../results/correlations' + file_name
    print("starting predictions for band " + band + " and regularization " + reg)
    accuracies = compute_benchmark(band, reg, params, epochs_combined=True,
                                   path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
    accuracies.to_csv(path_or_buf=save_path)
    print("accuracy table successfully saved for band" + band)
print("all accuracy tables successfully saved")


