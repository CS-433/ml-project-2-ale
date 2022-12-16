from plots import *


tables_path_RF = r'C:\Users\lucil\Desktop\RandomForest'
tables_path_SVM = r'C:\Users\lucil\Desktop\SVM_updated_GS'
tables_path_corr = r'C:\Users\lucil\Desktop\correlations'
saving_path = r'plots/'
y_scale_RF_all_epochs = np.linspace(0.35, 1, 14, endpoint=True)
y_scale_RF = np.linspace(0.8, 1, 9, endpoint=True)

# plot RF
plot_accuracies_wrt_parameter(tables_path_RF, saving_path, model_name="Random Forest", regularization="l2", 
                              save_fig=True, y_scale=y_scale_RF)
plot_accuracies_wrt_parameter(tables_path_RF, saving_path, model_name="Random Forest", regularization="log", 
                              save_fig=True, y_scale=y_scale_RF)
plot_accuracies_wrt_parameter_all_epochs(tables_path_RF, saving_path, "Random Forest", "l2", save_fig=True, 
                                         y_scale=y_scale_RF_all_epochs)
plot_accuracies_wrt_parameter_all_epochs(tables_path_RF, saving_path, "Random Forest", "log", save_fig=True, 
                                         y_scale=y_scale_RF_all_epochs)

# plot SVM
plot_accuracies_wrt_parameter(tables_path_SVM, saving_path, model_name="SVM", regularization="l2", save_fig=True)
plot_accuracies_wrt_parameter(tables_path_SVM, saving_path, model_name="SVM", regularization="log", save_fig=True)
plot_accuracies_wrt_parameter_all_epochs(tables_path_SVM, saving_path, "SVM", "l2", save_fig=True)
plot_accuracies_wrt_parameter_all_epochs(tables_path_SVM, saving_path, "SVM", "log", save_fig=True)

#plot benchmark
plot_accuracies_wrt_parameter(tables_path_corr, saving_path, "Correlations", "l2", save_fig=True, y_scale=y_scale_RF)

