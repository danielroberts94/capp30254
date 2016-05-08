# By Daniel Roberts
# Significant Parts of the Code below are structually borrowed from some of my Peers
# In particular, Michael Fosco. Structually, I found his code useful in organizing 
# my own thoughts about how to loop through the different parameters/models
# And some of the specific fixes Michael had proved invaluable to me where I would have been stuck 
# otherwise. As much as possible, I tried to tweak, and rewrite the code.

#I also borrow occasionally from Rayid Ghani's "Magic Loop" on github.
import math
import numpy as np 
import pandas as pd 
from scipy import optimize
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
import os, timeit, sys, itertools, re, time, requests, random, functools, logging, csv, datetime
from sklearn import linear_model, neighbors, ensemble, svm, preprocessing
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
''
criteriaHeader = ['AUC', 'Accuracy', 'Function called', 'Precision at .05',
                  'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5',
                  'Precision at .75','Precision at .85','Recall at .05','Recall at .10',
                  'Recall at .20','Recall at .25','Recall at .5','Recall at .75',
                  'Recall at .85','f1 at 0.05','f1 at 0.1','f1 at 0.2','f1 at 0.25',
                  'f1 at 0.5','f1 at 0.75','f1 at 0.85','test_time (sec)','train_time (sec)']

modelNames = ['LogisticRegression', 'KNeighborsClassifier', 'RandomForestClassifier', 'ExtraTreesClassifier',
              'AdaBoostClassifier', 'SVC', 'GradientBoostingClassifier', 'GaussianNB', 'DecisionTreeClassifier',
              'SGDClassifier']
depth = [10, 20, 50]
modelLR = {'model': LogisticRegression, 'solver': ['sag'], 'C' : [.01, .1, .5, 1],
          'class_weight': ['balanced', None],'tol' : [1e-5, 1e-3, 1], 'penalty': ['l2']} 
modelKNN = {'model': neighbors.KNeighborsClassifier, 'weights': ['uniform', 'distance'], 'n_neighbors' : [100, 500, 1000],
            'leaf_size': [60, 120],} 
modelRF  = {'model': RandomForestClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth, 'min_samples_split': [20, 50], 
            'bootstrap': [True],} 
modelET  = {'model': ExtraTreesClassifier, 'n_estimators': [25, 50, 100], 'criterion': ['gini', 'entropy'],
            'max_features': ['sqrt', 'log2'], 'max_depth': depth,
            'bootstrap': [True, False]}
modelAB  = {'model': AdaBoostClassifier, 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [5, 10, 25, 50, 100]}#, 200]}

modelSVM = {'model': svm.SVC, 'C':[0.1,1], 'max_iter': [1000, 2000], 'probability': [True], 
            'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}

modelNB  = {'model': GaussianNB}
modelDT  = {'model': DecisionTreeClassifier, 'criterion': ['gini', 'entropy'], 'max_depth': [10,20,50],
            'max_features': ['sqrt','log2'],'min_samples_split': [10, 20, 50]} 
modelSGD = {'model': SGDClassifier, 'loss': ['modified_huber', 'perceptron'], 'penalty': ['l1', 'l2', 'elasticnet'], }

modelList = [modelLR, modelKNN, modelRF, modelET, 
             modelAB, modelSVM, modelNB, modelDT,
             modelSGD] 

##################################################################################


# Function to read in csv data
def data_read(file_name):
    data = pd.read_csv(file_name)
    return data

#Function to explore data by printing out summary statistics and producing histograms for numerical
#data and bar charts for categorical data
'''
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
    if num_data.empty == False:
        for col in num_data:
            plot_hist(data[col], str(col), 20, dataset_name, 'hist')
            max_val = data[col].max()
            mean_val = data[col].mean()
            min_val = data[col].min()
            sd_val = data[col].std()
            if abs(max_val-mean_val)/(sd_val) >8 or abs(max_val-mean_val)/(sd_val) > 8:
                data[col] = data[col][data[col] > 0]
                log_vals = data[col].apply(np.log)
                plot_hist(log_vals, "log" + "_" + str(col), 20, dataset_name, 'hist')

    str_data = data.select_dtypes(include = ['object'])
    if str_data.empty == False:
        for col in str_data:
            plot_hist(data[col], str(col), 0, dataset_name, 'bar')

def plot_hist(df, title, num_bins, dataset_name, kind):
        plt.figure()
        if kind == 'hist':
            binwidth = ((df.max()- df.min())/num_bins)
            if binwidth > 1:
                binwidth = math.floor(binwidth)
            try:
                df.hist(bins=np.arange(df.min(), df.max() + binwidth, binwidth))
            except:
                plt.close()
                return
        elif kind == 'bar':
            df.value_counts().plot(kind='bar')
        plt.title(title)
        file_path = 'graphics/' + dataset_name + "_" + title + '.png'
        plt.savefig(file_path)
        plt.close()
    '''

#Simple function to fill variable missing vals with mean or mode of the def
def mean_mode_fill(data):
    num_data = data
    str_data = data
    if len(num_data.columns) > 0:
        data.update(num_data.fillna(num_data.mean()))
    #if len(str_data.columns) > 0:
     #   data.update(str_data.fillna(num_data.mode()[0]))
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
def build_classifier(x_var, y_var, k, model_dict):
    params = param_permutes(model_dict)
    model_num = len(params)
    print model_num
    results = []
    folds = cross_validation.KFold(len(y_var), k)
    j = 0
    for pars in params:
        clf = model_dict['model'](**pars)
        train_times = [None]*k
        pred_probs = [None]*k
        test_times = [None]*k
        y_tests = [None]*k
        accs = [None]*k
        #try:
        i = 0
        for train, test in folds:
            #if i > 1:
            #    break
            print str(i) + "beginning the olfy loop!"
            x_train, x_test = x_var._slice(train, 0), x_var._slice(test, 0)
            y_train, y_test = y_var._slice(train, 0), y_var._slice(test, 0)
            y_tests[i] = y_test
            train_start = time()
            model_fit = clf.fit(x_train, y_train)
            time()
            train_time = time() - train_start
            train_times[i] = train_time
            test_start = time()
            pred_prob = model_fit.predict_proba(x_test)[:,1]
            test_time = time() - test_start
            test_times[i] = test_time
            pred_probs[i] = pred_prob
            accs[i] = model_fit.score(x_test,y_test)
            i += 1 
            print i
        #except:
        #    print "Parameter Values Invalid"
        #    continue
        #if i > 1:
        #    break
        evals = evaluation(y_tests, pred_probs, train_times, test_times, accs, str(clf))
        results.append(evals)
        print "after " + str(j)
        j += 1
    # THIS IS A CRITERIA THAT CAN BE CHANGED DEPENDING ON THE DESIRED RESULTS
    # IN THIS CASE I USE AUC.
    # Joint Maximum, Joint Minima, and a variety of other combinations of features can be used to subset here to find
    # the best parameter set in a given model. This function call can be ommitted to output all possible parameter sets.
    print results
    results = maxim_param('AUC', results)
    return results

def maxim_param(param_name, results):
    param = [mod[param_name] for mod in results if mod != None]
    max_par = max(param)
    no_nones =  [x for x in results if x != None]
    new_res =  [x for x in no_nones if x[param_name] == max_par]
    return new_res

def evaluation(y_test, pred_probs, train_times, test_times, accuracies, model_name):
    levels = ['Precision at .05', 'Precision at .10', 'Precision at .2', 'Precision at .25', 'Precision at .5', 'Precision at .75', 'Precision at .85']
    recalls = ['Recall at .05', 'Recall at .10', 'Recall at .20', 'Recall at .25', 'Recall at .5', 'Recall at .75', 'Recall at .85']
    levels= [.05, .1, .2, .25, .5, .75, .85]
    levnums = len(levels)
    results = {}
    num_folds = len(y_test)
    folds = range(0, num_folds)
    results['Model'] = model_name
    for x in xrange(0, levnums):
        threshold = levels[x]
        y_pred = [0]*num_folds
        for j in folds:
            y_pred[j] = np.asarray([1 if i >= threshold else 0 for i in pred_probs[j]])
        prec = [metrics.precision_score(y_test[j], y_pred[j]) for j in folds]
        rec = [metrics.recall_score(y_test[j], y_pred[j]) for j in folds]
        precStd = np.std(prec)
        recStd = np.std(rec)

        f1S = [2*(prec[j]*rec[j])/(prec[j]+rec[j]) for j in folds]
        f1Std = np.std(f1S)

        precM = np.mean(prec)
        recM = np.mean(rec)
        f1M = np.mean(f1S)
        results[levels[x]] =  str(precM) + ' (' + str(precStd) + ')'
        results[recalls[x]] =  str(recM) + ' (' + str(recStd) + ')'
        results['f1 at ' + str(threshold)] =  str(f1M) + ' (' + str(f1Std) + ')'

    auc = [metrics.roc_auc_score(y_test[j], pred_probs[j]) for j in folds]
    aucStd = np.std(auc)
    aucM = np.mean(auc)
    trainM = np.mean(train_times)
    trainStd = np.std(train_times)
    testM = np.mean(test_times)
    testStd = np.std(test_times)
    accM = np.mean(accuracies)
    accStd = np.std(accuracies)

    results['AUC'] = str(aucM) + ' (' + str(aucStd) + ')'
    results['train_time (sec)'] = str(trainM) + ' (' + str(trainStd) + ')'
    results['test_time (sec)'] = str(testM) + ' (' + str(testStd) + ')'
    results['Accuracy'] = str(accM) + ' (' + str(accStd) + ')'

    return results

def removeKey(d, key):
    r = dict(d)
    del r[key]
    return r

def param_permutes(model_dict):
    params = []
    just_params = removeKey(model_dict, 'model')
    keys  = just_params.keys()
    all_together = [just_params[x] for x in keys]
    param_cross = list(itertools.product(*all_together))
    cross_length = len(param_cross)
    params = [0] * cross_length
    for i in range(0, cross_length):
        params[i] = dict(zip(keys, param_cross[i]))
    return params

def kfolds_train_only_pipe(data, y_name, folds, models = modelList, fill = mean_mode_fill):
    data = mean_mode_fill(data)
    y_var = data[y_name]
    x_var = data.drop(y_name, 1)
    results = []
    for mod in models:
        mod_result = build_classifier(x_var, y_var, folds, mod)
        results += mod_result
    return results

def make_header(output_dic):
    header = None
    spot = 0
    limit = len(output_dic)
    while(header == None and spot < limit):
        if output_dic[spot] != None:
            header = [x for x in output_dic[spot].keys()]

    if header == None:
        raise Exception('Unable to grab appropriate header. Please check pipeLine')

    header.sort()
    return header

def format_data(header, output_dic):
    length = len(output_dic)
    format = [[]] * length
    lenMH = len(header)

    outdex = 0
    index = 0
    for x in output_dic:
        temp = [None] * lenMH
        for j in header:
            temp[index] = x[j]
            index += 1
        index = 0
        format[index] = temp
        outdex += 1
    return format

def write_output(file_name, output_dic):
    header = make_header(output_dic)
    fin = format_data(header, output_dic)
    fin.insert(0, header)
    with open(file_name, "w") as fout:
        writer = csv.writer(fout)
        for f in fin:
            writer.writerow(f)
        fout.close()
