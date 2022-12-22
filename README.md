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
    ├── cross_validation.py
    ├── helpers.py
    ├── implementations.py
    ├── model_selection.py
    ├── plots
    ├── plots.py
    ├── README.md
    ├── run.py
    └── Project2.pdf



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


Python files:
---------

filename                        | description
--------------------------------|------------------------------------------
cross_validation.py             |Set of functions that perform the k_fold-cross validation of the different prediction models from implementations.py
helpers.py                      |Set of useful functions used throughout the project
implementations.py              |Data preprocessing function as well as our model prediction functions. These include Least Squares, Ridge Regression, Gradient                                      Descend, Stochastic Gradient Descend, Polynomial Regression, Logistic Regression and Regularized Logistic Regression
model_selection.py              |File containing the resulting MSE computed by different hyperparameters.
plots.py                        |Functions used to plot some of our results
run.py                          |File that runs a our best model and creates the submission file for AICrowd

Authors
=======
Emilie MARCOU, Lucille NIEDERHAUSER & Anabel SALAZAR DORADO
