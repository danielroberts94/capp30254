# By Daniel Roberts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

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