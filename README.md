Machine Learning Project: Brain Fingerprint from the brain activity
==========
In partnership with the Medical Image Processing Lab (MIP:Lab)

Supervisors: Hamid Behjat and Ekansh Sareen


Overview
========
It has previously been shown that it is possible to identify individuals using brain activity signals such as functional Magnetic Resonance Imaging (fMRI) or magnetoencephalography (MEG), a process known as brain fingerprinting. Until now, this has always been done by computing statistical brain region dependencies to build functional connectivity (FC) matrices. Here we explore a new approach of brain fingerprinting using MEG data that differs from previous methods in two ways. First, we use the brain signals to learn brain graphs instead of computing FC matrices. Secondly, we explore the use of machine learning classifiers to identify individuals instead of computing inter-subjects correlations. <br/>
<br/>


Directory layout
================

    Directory                           # Main directory
    
    ├── data_sets
            ├────── Test_set
            └─────── Train_set    
    ├── plots
            ├────── SVM_l2_all_epochs_accuracies.png
            └─────── SVM_log_all_epochs__accuracies.png 
    ├── utils
            ├────── HCP_info
                          └────── MEG84_subjects_ID.mat
            └────── gspbox                                  
    ├── Data_Visualisation.ipynb
    ├── README.md
    ├── create_datasets.py
    ├── helpers.py
    ├── plots.py
    ├── models.py
    ├── run_Random_Forest.py
    ├── run_Random_Forest_all_epochs.py
    ├── run_SVM.py
    ├── run_SVM_all_epochs.py
    ├── run_SVM_without_var.py
    ├── run_correlations.py
    ├── run_correlation_all_epochs.py
    ├── run_sparsity.py
    ├── scr_learn_graph_LEA_allsubjs.m    
    └── Report.pdf
     



Description of contents
==============

Directories:
---------
Directory name                  | description
--------------------------------|------------------------------------------
plots           			    |Contains all the plots of the preliminary data visualization and of the tuning of hyper-parameters


Non Python files:
-----------

filename                        | description
--------------------------------|------------------------------------------
README.md                       | Text file (markdown format) describing all the files of the project
Data_Visualisation.ipynb        | 
scr_learn_graph_LEA_allsubjs.m  |
Report.pdf                      |





Python files:
---------

filename                        | description
--------------------------------|------------------------------------------
create_datasets.py              |
helpers.py                      |Set of useful functions used throughout the project
plots.py                        |
models.py                       |
run_Random_Forest.py            |
run_Random_Forest_all_epochs.py |
run_SVM.py                      |
run_SVM_all_epochs.py           |
run_SVM_without_var.py          |
run_correlations.py             |
run_correlation_all_epochs.py   |
run_sparsity.py                 |

Authors
=======
Emilie MARCOU, Lucille NIEDERHAUSER & Anabel SALAZAR DORADO
