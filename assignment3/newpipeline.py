# By Daniel Roberts


import numpy as np 
import pandas as pd 
from scipy import optimize
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
import multiprocessing as mp
from multiprocessing import Process, Queue
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re, time, requests, random, functools, logging, csv, datetime
import seaborn as sns
from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
from numba.decorators import jit, autojit
from numba import double #(nopython = True, cache = True, nogil = True [to run concurrently])
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from time import time 

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
def build_classifier(X_var, y_var, model_dict):
    params = makeDicts(model_dict)
    model_num = len(params)
    results = [None]*total
    folds = cross_validation.KFold(len(y_var), k)

    z = 0
    for pars in params:
        clf = fun_build(model_dict['model'], pars)
        i = 0
        for train, test in kf:
            x_train, x_test = X._slice(train, 0), X._slice(test, 0)
            y_train, y_test = y._slice(train, 0), y._slice(test, 0)
            yTests[i] - yTest

            train_start = time()
            model_fit = clf.fit(x_train,y_train)
            train_time = time() - train_start
            train_times[i] = train_time
            test_start = time()



            
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
def evaluation(y_test, predProbs, train_times, test_times, accuracies, model_name):
    prec_giv_threshhold = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
    recall_giv_threshhold = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
    levels= [.05, .1, .2, .25, .5, .75, .85]
    levnums = len(levels)
    results = {}
    num_folds = len(y_test)
    folds = range(0, num_folds)
    results['Model'] = model_name
    for x in xrange(0, tots):
        threshhold = amts[x]
        prec = [precision_at_k(y_test[j], predProbs[j], threshhold)  for j in folds]
    auc = [metrics.roc_auc_score(yTests[j], predProbs[j]) for j in folds]
    aucStd = np.std(auc)
    aucM = np.mean(auc)
    trainM = np.mean(train_times)
    trainStd = np.std(train_times)
    testM = np.mean(test_times)
    testStd = np.std(test_times)
    accM = np.mean(accuracies)
    accStd = np.std(accuracies)

    res['AUC'] = makeResultString(aucM, aucStd)
    res['train_time (sec)'] = makeResultString(trainM, trainStd)
    res['test_time (sec)'] = makeResultString(testM, testStd)
    res['Accuracy'] = makeResultString(accM, accStd)

    return res


def fun_build(name, args):
    res_fun = name(**args)
    return res_fun

def param_permutes(model_dict):
    params = []
    just_params = removeKey(d, 'model')
    keys  = just_params.keys()
    all_together = [just_params[x] for x in keys]
    param_cross = list(itertools.product(*l))
    cross_length = len(param_cross)
    params = [0] * cross_length
    for i in range(0, cross_length):
        params[i] = dict(zip(keys, combos[i]))
    return params

#Borrowed from Rayid
def precision_at_k(y_true, y_scores, k):
    threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))]
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)


def clf_predict(clf, test_data, feat_names):
    test_feats = test_data[feat_names]
    
    pred_values = clf.predict(test_feats)
    return pred_values

#Determine accuracy from a given classifier
def clf_accuracy(clf, predictors, predicted):
    return clf.score(predictors, predicted)