Machine Learning Project: Brain Fingerprint from the brain activity
==========
In partnership with the Medical Image Processing Lab (MIP:Lab)

Supervisors: Hamid Behjat and Ekansh Sareen


Overview
========
This machine learning project helps to predict the identity of a individu given its brain activity. To do so, we learn a graph from MEG (Magnetoencephalography) data to extract features use in our classifier. We started with a basic multi class logistic classifier before trying more complex ones. Moreover, we tuned our learning graph method to obtain the best features and so the best accuracy. <br/>
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
