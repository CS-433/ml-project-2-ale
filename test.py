import numpy as np
from models import *
from helpers import *
import scipy.io

"""
data_path = r'data'
matrix, y = load_csv_data(data_path)

print("Matrix:", matrix)
print(matrix.shape)
print("Y labels", y)
"""

"""
f = h5py.File('MEG84_subjects_ID.mat', 'r')
data = f.get('data/IDs')
data = np.array(data)  # For converting to a NumPy array
print(data)
"""
'''
mat = scipy.io.loadmat('MEG84_subjects_ID.mat')
print(mat['IDs'])
'''
'''
# How to get all the files names in all subdirectories:
filelist = []

for root, dirs, files in os.walk('data'):
    for file in files:
        # append the file name to the list
	    filelist.append(os.path.join(file))

# print all the file names
for name in filelist:
    print(name)
'''
'''
# How to load all files in all subdirectories:
files = glob.glob('data/**/*.mat', recursive=True)
for file in files:
    print(file)
'''
'''
file = scipy.io.loadmat('data/102816/102816.3-Restin.beta.sigformat_env.distance_gspbox.alphas_20.betas_20.noDownsampling.mat', appendmat=0)
results = file['RESULTS']
ID = results["ID"]
wl2 = results["W_l2"]
wlog = results["W_log"]
session = results["session"]
alphas = results["alphas"]
size_wl2 = results["W_l2_size"]

#size_wl2 = size_wl2.item()
#size_wl2 = np.squeeze((size_wl2))

wl2 = wl2[0][0]
wlog = wlog[0][0]
alphas = alphas[0][0].T
print(alphas)
print(alphas.shape[0])
print(wlog.shape)
print(wlog.shape[0])
print(wlog.shape[1])
print(wlog.shape[2])
print(wlog.shape[3])
'''

'''
# Test of new load data
# final_data_set_train_wl2, final_data_set_train_wlog, final_data_set_test_wlog, final_data_set_test_wl2 = load_csv_data_updated('MATLAB/learned_graphs')
final_data_set_train_wl2_alpha, final_data_set_train_wl2_beta, final_data_set_train_wl2_delta, \
    final_data_set_train_wl2_gamma, final_data_set_train_wl2_theta, final_data_set_test_wl2_alpha, \
    final_data_set_test_wl2_beta, final_data_set_test_wl2_delta, final_data_set_test_wl2_gamma, \
    final_data_set_test_wl2_theta, final_data_set_train_wlog_alpha, final_data_set_train_wlog_beta, \
    final_data_set_train_wlog_delta, final_data_set_train_wlog_gamma, final_data_set_train_wlog_theta,\
    final_data_set_test_wlog_alpha, final_data_set_test_wlog_beta, final_data_set_test_wlog_delta, \
    final_data_set_test_wlog_gamma, final_data_set_test_wlog_theta = load_csv_data_updated('data')

print("wl2 train alpha:", len(final_data_set_train_wl2_alpha))
print(final_data_set_train_wl2_alpha[0].shape)
print("wlog test theta:", len(final_data_set_test_wlog_theta))
print(final_data_set_test_wlog_theta[0].shape) '''

#size_wl2 = size_wl2.item()
#size_wl2 = np.squeeze((size_wl2))

wl2 = wl2[0][0]
wlog = wlog[0][0]
alphas = alphas[0][0].T
band = results["band"]
band = np.squeeze(band[0][0])
print(band)
'''
print(alphas.shape[0])
print(wlog.shape)
print(wlog.shape[0])
print(wlog.shape[1])
print(wlog.shape[2])
print(wlog.shape[3])
'''
'''print("wlog train:", len(final_data_set_train_wlog))
print(final_data_set_train_wlog[0].shape)
print("wl2 test:", len(final_data_set_test_wl2))
print(final_data_set_test_wl2[0].shape)
print("wlog test:", len(final_data_set_test_wlog))
print(final_data_set_test_wlog[0].shape)'''
# we have a list of size 20. Each element i of the list should be a matrix 360 (24*3*5) x 10878

# print(final_data_set_train_wl2)
'''

file = scipy.io.loadmat('data/102816/102816.3-Restin.beta.sigformat_env.distance_gspbox.alphas_20.betas_20.noDownsampling.mat', appendmat=0)
results = file['RESULTS']
ID = results["ID"]
wl2 = results["W_l2"]
wlog = results["W_log"]
session = results["session"]
alphas = results["alphas"]
size_wl2 = results["W_l2_size"]

###### Script used to test the SVM model on the servers ######

# x_train, y_train = load_data_set("alpha", "wl2", "train", 0.9, path=r'../data_sets/')
# x_test, y_test = load_data_set("alpha", "wl2", "test", 0.9, path=r'../data_sets/')
x_train, y_train = load_data_set("alpha", "wl2", "train", 0.5, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
x_test, y_test = load_data_set("alpha", "wl2", "test", 0.5, path=r'../../Data3/Hamid_ML4Science_ALE/data_sets/')
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

title = "confusion_matrix_wl2_alpha_0.5"
# accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_fig=True, title=title,
#                                                       save_path='plots/')
accuracy, C, gamma, kernel = SVM_tune_predict_evaluate(x_train, y_train, x_test, y_test, save_fig=True, title=title,
                                                       save_path='../../Data3/Hamid_ML4Science_ALE/plots/')
print(accuracy)
print(C)
print(gamma)
print(kernel)
