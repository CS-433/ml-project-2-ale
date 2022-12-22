from helpers import *
from plots import *

path_of_data = r'Sample_data'
save_path_of_CSV_per_epoch = r'../results/sparsity_csv/all_epochs/'
save_path_of_CSV_all_epoch = r'../results/sparsity_csv/per_epoch/'

save_path_of_sparsity_plots = r'plots/'

# make the csvs 
save_sparsities_each_epochs(path_of_data, save_path_of_CSV_per_epoch)
save_sparsities_all_epochs(path_of_data, save_path_of_CSV_all_epoch)

# make the plots
plot_sparsities_per_epoch(save_path_of_CSV_per_epoch, save_path_of_sparsity_plots, save_fig=True)
plot_sparsities_all_epochs(save_path_of_CSV_all_epoch, save_path_of_sparsity_plots, save_fig=True)
