import numpy as np

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


# Test of new load data
final_data_set_train_wl2, final_data_set_train_wlog, final_data_set_test_wlog, final_data_set_test_wl2 = load_csv_data_updated('data')
print("wl2 train:", final_data_set_train_wl2.shape)
print("wlog train:", final_data_set_train_wlog.shape)
print("wl2 test:", final_data_set_test_wl2.shape)
print("wlog test:", final_data_set_train_wlog.shape)

print(final_data_set_train_wl2)