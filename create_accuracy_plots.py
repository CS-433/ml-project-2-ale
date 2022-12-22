from plots import *

# path to the different results directory
tables_path_RF = r'../results/RandomForest'
tables_path_SVM = r'../SVM'
tables_path_corr = r'../results/correlations'
tables_path_all = r'../results'
saving_path = r'plots/'
# different y scales used for the plots
y_scale_RF_all_epochs = np.linspace(0.35, 1, 14, endpoint=True)
y_scale_RF = np.linspace(0.8, 1, 9, endpoint=True)
# defining parameters and bands used in our plots
bands = ["alpha", "beta", "delta", "gamma", "theta"]
params = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.6499999999999999, 0.7, 0.75, 0.8, 0.85,
          0.9, 0.95, 1.]


# plot RF
plot_accuracies_wrt_parameter(tables_path_RF, saving_path, model_name="Random Forest", regularization="l2", 
                              save_fig=False, y_scale=y_scale_RF, params=params)
plot_accuracies_wrt_parameter(tables_path_RF, saving_path, model_name="Random Forest", regularization="log", 
                              save_fig=False, y_scale=y_scale_RF, params=params)
plot_accuracies_wrt_parameter_from1file(tables_path_RF, saving_path, "Random Forest", "l2", save_fig=False,
                                         y_scale=y_scale_RF_all_epochs, params=params)
plot_accuracies_wrt_parameter_from1file(tables_path_RF, saving_path, "Random Forest", "log", save_fig=False,
                                         y_scale=y_scale_RF_all_epochs, params=params)

# plot SVM
plot_accuracies_wrt_parameter(tables_path_SVM, saving_path, model_name="SVM", regularization="l2", save_fig=False,
                              params=params)
plot_accuracies_wrt_parameter(tables_path_SVM, saving_path, model_name="SVM", regularization="log", save_fig=False,
                              params=params)
plot_accuracies_wrt_parameter_from1file(tables_path_SVM, saving_path, "SVM", "l2", save_fig=False, params=params)
plot_accuracies_wrt_parameter_from1file(tables_path_SVM, saving_path, "SVM", "log", save_fig=False, params=params)

# plot benchmark
plot_accuracies_wrt_parameter(tables_path_corr, saving_path, "Correlations", "l2", save_fig=False, y_scale=y_scale_RF,
                              params=params)
plot_accuracies_wrt_parameter(tables_path_corr, saving_path, "Correlations", "log", save_fig=False, y_scale=y_scale_RF,
                              params=params)
plot_accuracies_wrt_parameter(tables_path_corr, saving_path, "Correlations_all_epochs", "l2", save_fig=False,
                              y_scale=y_scale_RF_all_epochs, all_epochs=True, params=params)
plot_accuracies_wrt_parameter(tables_path_corr, saving_path, "Correlations_all_epochs", "log", save_fig=False,
                              y_scale=y_scale_RF_all_epochs, all_epochs=True, params=params)

# plot all models per band
for band in bands:
    plot_accuracies_all_settings_1band(tables_path_all, saving_path, band, y_scale=y_scale_RF, save_fig=False,
                                       params=params)

