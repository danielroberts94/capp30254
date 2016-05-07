#Daniel Roberts Assignment 2

import pipeline as pp
import numpy as np

test_data = pp.data_read('data_input/cs-test.csv')
train_data = pp.data_read('data_input/cs-training.csv')

pp.data_explore(test_data, "test_set")
pp.data_explore(train_data, "train_set")

pp.kfolds_train_only_pipe(test_data, 'SeriousDlqin2yrs', 5)
# Attempt Three different classifiers, find accuracies and predicted values