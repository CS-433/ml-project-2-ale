from helpers import *
from plots import *

# change paths
path_of_data = r'../../Data3/Hamid_ML4Science_ALE/MATLAB/learned_graphs'
save_path_of_CSV_per_epoch = r'../../Data3/Hamid_ML4Science_ALE/sparsity_csv/all_epochs/'
save_path_of_CSV_all_epoch = r'../../Data3/Hamid_ML4Science_ALE/sparsity_csv/per_epoch/'

save_path_of_sparsity_plots = r'../../Data3/Hamid_ML4Science_ALE/sparsity/plots/'

print("finding lowest number of epochs")
path = os.path.join(path_of_data, '**/*.mat')
files = glob.glob(path, recursive=True)
lowest_l2 = 0
lowest_log = 0
for file in files:
    session, alphas, betas, id, frequency_band, size_wl2, size_wlog, wl2, wlog = load_mat_file(file)
    print(size_wl2[3])
    if size_wl2[3] < lowest_l2:
        lowest_l2 = size_wl2[3]
    print(size_wlog[3])
    if size_wlog[3] < lowest_log:
        lowest_log = size_wlog[3]

print("lowest l2 = ", lowest_l2)
print("lowest log =", lowest_log)

'''
# make the csvs 
save_sparsities_each_epochs(path_of_data, save_path_of_CSV_per_epoch)
save_sparsities_all_epochs(path_of_data, save_path_of_CSV_all_epoch)

# make the plots
plot_sparsities_per_epoch(save_path_of_CSV_per_epoch, save_path_of_sparsity_plots, save_fig=True)
plot_sparsities_all_epochs(save_path_of_CSV_all_epoch, save_path_of_sparsity_plots, save_fig=True)
'''