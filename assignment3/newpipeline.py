# By Daniel Roberts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

##################################################################################
'''
List of models and parameters
'''

cores = mp.cpu_count()-1
modelLR = {'model': LogisticRegression, 'solver': ['liblinear'], 'C' : [.01, .1, .5, 1, 5, 10, 25],
          'class_weight': ['balanced', None], 'n_jobs' : [cores],
          'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'penalty': ['l1', 'l2']}
modelLSVC = {'model': svm.LinearSVC, 'tol' : [1e-7, 1e-5, 1e-4, 1e-3, 1e-1, 1], 'class_weight': ['balanced', None],
             'max_iter': [1000, 2000], 'C' :[.01, .1, .5, 1, 5, 10, 25]}
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [2, 5, 10, 50, 100, 500, 1000, 10000],
            'leaf_size': [15, 30, 60, 120], 'n_jobs': [cores]}
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [5, 10, 25, 50, 100, 200, 1000, 10000], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': [1, 5, 10, 20, 50, 100], 'min_samples_split': [2, 5, 10, 20, 50],
            'bootstrap': [True, False], 'n_jobs':[cores]}
modelET  = {'model': ExtraTreesClassifier, 'n_estimatores': [5, 10, 25, 50, 100, 200, 1000, 10000], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': [1, 5, 10, 20, 50, 100],
            'bootstrap': [True, False], 'n_jobs':[cores]}
#base classifier for adaboost is automatically a decision tree
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1, 10, 100, 200, 1000, 10000]}
modelSVM = {'model': svm.SVC, 'C':[0.00001,0.0001,0.001,0.01,0.1,1,10], 'probability': [True], 'kernel': ['rbf', 'poly', 'sigmoid']}
modelGB  = {'model': GradientBoostingClassifier, 'learning_rate': [0.001,0.01,0.05,0.1,0.5], 'n_estimators': [1,10,100], #200,1000,10000], these other numbers took way too long to calc
            'max_depth': [1,3,5,10,20,50,100], 'subsample' : [0.1, .2, 0.5, 1.0]}
#Naive Bayes below
modelNB  = {'model': GaussianNB}
modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 
            'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10, 20, 50]}
modelSGD = {'model': SGDClassifier, 'loss': ['modified_huber', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], 
            'n_jobs': [cores]}

modelList = [modelLR, modelLSVC, modelKNN, modelRF, modelET, 
             modelAB, modelSVM, modelGB, modelNB, modelDT,
             modelSGD]

##################################################################################


# Function to read in csv data
def data_read(file_name):
    data = pd.read_csv(file_name)
    return data

#Function to explore data by printing out summary statistics and producing histograms for numerical
#data and bar charts for categorical data
def data_explore(data, dataset_name):
    stats = data.describe()
    print stats
    print stats.to_latex()
    counts = data.count()
    missings = np.multiply(-1, np.subtract(counts, len(data.index)))
    print missings
    modes = data.mode(axis=0)
    print modes
    num_data = data.select_dtypes(include = ['number'])
    for col in num_data:
        plt.figure()
        try:
            data[col].hist(bins=100)
            plt.title(col)
            file_path = 'graphics/' + dataset_name + "_" + str(col) + '.png'
            plt.savefig(file_path)
            plt.close()
        except:
            plt.close()
            continue
    str_data = data.select_dtypes(include = ['object'])
    for col in str_data:
        plt.figure()
        try:
            data[col].value_counts().plot(kind='bar')
            plt.title(col)
            file_path = 'graphics/' + dataset_name + "_" + str(col) + '.png'
            plt.savefig(file_path)
            plt.close()
        except:
            plt.close()
            continue
#Simple function to fill variable missing vals with mean or mode of the variables
def mean_mode_fill(data):
    num_data = data.select_dtypes(include = ['number'])
    str_data = data.select_dtypes(include = ['object'])
    if len(num_data.columns) > 0:
        data.update(num_data.fillna(num_data.mean()))
    if len(str_data.columns) > 0:
        data.update(str_data.fillna(num_data.mode()[0]))
    data.to_csv("data_output/fill_output.csv")
    return data

#function to discretize a numerical variable into a set number of quantiles.
def data_discretize(data, col_names, quantiles):
    for col, quant in zip(col_names,quantiles):
        data.update(pd.qcut(data[col], q = quant))
    return data

#function to create a series of binary variables from a categorical variable
#aka "Dummy Variables"
def data_dummify(data, col_names, quantiles):
    data = pd.get_dummies(data, columns = col_names)
    return data

#Builds one of three types of classifiers at present. Future versions will be amenable to parameter choice
def build_classifier(train_data, learnvar, feat_names, clf_name):
    train_feats = train_data[feat_names]
    train_learnvar = train_data[learnvar]
    clfs = {'log': LogisticRegression,
            'Kneighbors': KNeighborsClassifier,
            'LSVC': LinearSVC}
    if len(train_feats.shape) < 2:
        train_feats.reshape(len(train_feats),1)
    if len(train_learnvar.shape) < 2:
        train_feats.reshape(len(train_learnvar),1)
    clf = clfs[clf_name]()
    clf.fit(train_feats,train_learnvar)
    return clf
#Predict values based on a given classifer (note: not classifier name)
def clf_predict(clf, test_data, feat_names):
    test_feats = test_data[feat_names]
    
    pred_values = clf.predict(test_feats)
    return pred_values

#Determine accuracy from a given classifier
def clf_accuracy(clf, predictors, predicted):
    return clf.score(predictors, predicted)