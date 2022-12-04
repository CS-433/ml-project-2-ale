from helpers import *

# create all the data sets from the matrix returned by the learn graph script
final_data_set_train_wl2_alpha, final_data_set_train_wl2_beta, final_data_set_train_wl2_delta, \
    final_data_set_train_wl2_gamma, final_data_set_train_wl2_theta, final_data_set_test_wl2_alpha, \
    final_data_set_test_wl2_beta, final_data_set_test_wl2_delta, final_data_set_test_wl2_gamma, \
    final_data_set_test_wl2_theta, final_data_set_train_wlog_alpha, final_data_set_train_wlog_beta, \
    final_data_set_train_wlog_delta, final_data_set_train_wlog_gamma, final_data_set_train_wlog_theta,\
    final_data_set_test_wlog_alpha, final_data_set_test_wlog_beta, final_data_set_test_wlog_delta, \
    final_data_set_test_wlog_gamma, final_data_set_test_wlog_theta = create_all_sets('../../Data3/Hamid_ML4Science_ALE/MATLAB/learned_graphs')

print("data sets created")
