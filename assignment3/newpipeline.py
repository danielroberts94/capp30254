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
    if num_data.empty == False
        for col in num_data:
            if col != ''
            plot_hist(data[col], str(col), 20, dataset_name, 'hist')
            max_val = data[col].max()
            tenth_val = data[col].quan
            mean_val = data[col].mean()
            min_val = data[col].min()
            sd_val = data[col].std()
            if abs(max_val-mean_val)/(sd_val) || abs(max_val-mean_val)/(sd_val)  > 8:
                log_vals = data[col].apply(log)
                plot_hist(log_vals, "log" + "_" str(col), 20, dataset_name, 'hist')

    str_data = data.select_dtypes(include = ['object'])
    if str_data.empty == False
        for col in str_data:
            plot_hist(data[col], str(col), 0, dataset_name, 'bar')

def plot_hist(df, title, num_bins, dataset_name, kind):
        plt.figure()
        if kind == 'hist':
            binwidth = floor(df.max()- df.min())/num_bins
            df.hist(bins=np.arange(min(df.min()), max(f.max()) + binwidth, binwidth))
        elif kind == 'bar':
            df.value_counts().plot(kind='bar')
        plt.title(title)
        file_path = 'graphics/' + dataset_name + "_" + title + '.png'
        plt.savefig(file_path)
        plt.close()

#Simple function to fill variable missing vals with mean or mode of the def
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
def build_classifier(X_var, y_var, k, model_dict):
    params = param_permutes(model_dict)
    model_num = len(params)
    results = [None]*total
    folds = cross_validation.KFold(len(y_var), k)

    j = 0
    for pars in params:
        clf = fun_build(model_dict['model'], pars)
        i = 0
        train_times = [None]*k
        pred_probs = [None]*k
        test_times = [None]*k
        y_tests = [None]*k
        accs = [None]*k

        for train, test in kf:
            x_train, x_test = X._slice(train, 0), X._slice(test, 0)
            y_train, y_test = y._slice(train, 0), y._slice(test, 0)
            y_tests[i] - yTest

            train_start = time()
            model_fit = clf.fit(x_train,y_train)
            train_time = time() - train_start
            train_times[i] = train_time
            test_start = time()
            pred_prob = model_fit.predict_proba(x_test)[:,1]
            test_time = time() - test_start
            test_times[i] = test_time
            pred_probs[i] = pred_prob
            accs[i] = model_fit.score(XTest,yTest)
            i += 1
        evals = evaluation(y_tests, pred_probs, train_times, test_times, accs, str(clf))
        results[j] = evals
        j += 1
    return results


def evaluation(y_test, pred_probs, train_times, test_times, accuracies, model_name):
    prec_giv_threshhold = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
    recall_giv_threshhold = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
    levels= [.05, .1, .2, .25, .5, .75, .85]
    levnums = len(levels)
    results = {}
    num_folds = len(y_test)
    folds = range(0, num_folds)
    results['Model'] = model_name
    for x in xrange(0, tots):
        threshhold = levels[x]
        y_pred = np.asarray([1 if i >= threshold else 0 for i in pred_probs])
        prec = [metrics.precision_score(y_test[j], y_pred) for j in folds]
        rec = [metrics.recall_score(y_test[j], y_pred) for j in folds]
        precStd = np.std(prec)
        recStd = np.std(rec)

        f1S = [2*(prec[j]*rec[j])/(prec[j]+rec[j]) for j in folds]
        f1Std = np.std(f1S)

        precM = np.mean(prec)
        recM = np.mean(rec)
        f1M = np.mean(f1S)
        results[levels[x]] = makeResultString(precM, precStd)
        results[recalls[x]] = makeResultString(recM, recStd)
        results['f1 at ' + str(thresh)] = makeResultString(f1M, f1Std)

    auc = [metrics.roc_auc_score(y_test[j], predProbs[j]) for j in folds]
    aucStd = np.std(auc)
    aucM = np.mean(auc)
    trainM = np.mean(train_times)
    trainStd = np.std(train_times)
    testM = np.mean(test_times)
    testStd = np.std(test_times)
    accM = np.mean(accuracies)
    accStd = np.std(accuracies)

    results['AUC'] = makeResultString(aucM, aucStd)
    results['train_time (sec)'] = makeResultString(trainM, trainStd)
    results['test_time (sec)'] = makeResultString(testM, testStd)
    results['Accuracy'] = makeResultString(accM, accStd)

    return results

def makeResultString(mean, std):
    return str(mean) + ' (' + str(std) + ')' 


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
def precision_at_thresh(y_true, y_scores, thresh):
    y_pred = np.asarray([1 if i >= k else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)

def precision_at_k(y_true, y_scores, k):
    y_pred = np.asarray([1 if i >= k else 0 for i in y_scores])
    return metrics.precision_score(y_true, y_pred)

def clf_predict(clf, test_data, feat_names):
    test_feats = test_data[feat_names]
    
    pred_values = clf.predict(test_feats)
    return pred_values

#Determine accuracy from a given classifier
def clf_accuracy(clf, predictors, predicted):
    return clf.score(predictors, predicted)

def kfolds_train_only_pipe(data, models = modelList, y_name, folds, fill = mean_mode_fill):
    data = mean_mode_fill(data)
    y_var = data[yName]
    x_var = data.drop(yName, 1)
    results = []
    for mod in models:
        mod_result = build_classifier(x_var, y_var, k, mod)
        results += mod_result
    return results

def plot_precision_recall_n(y_true, y_prob, model_name):
    from sklearn.metrics import precision_recall_curve
    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    #plt.savefig(name)
    plt.show()

def makeHeader(d):
    header = None
    spot = 0
    limit = len(d)
    while(header == None and spot < limit):
        if d[spot] != None:
            header = [x for x in d[spot].keys()]

    if header == None:
        raise Exception('Unable to grab appropriate header. Please check pipeLine')

    header.sort()
    return header

def writeResultsToFile(fName, d):
    header = makeHeader(d)
    fin = formatData(header, d)
    fin.insert(0, header)
    try:
        with open(fName, "w") as fout:
            writer = csv.writer(fout)
            for f in fin:
                writer.writerow(f)
            fout.close()
    except:
        return -1
    return 0

