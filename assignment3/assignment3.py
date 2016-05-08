#Daniel Roberts Assignment 2

import newpipeline as pp
import numpy as np

#test_data = pp.data_read('data_input/cs-test.csv')
train_data = pp.data_read('data_input/cs-training.csv')

#pp.data_explore(test_data, "test_set")
#pp.data_explore(train_data, "train_set")

results = pp.kfolds_train_only_pipe(train_data, 'SeriousDlqin2yrs', 5)
pp.write_output("test.csv", results)
# Attempt Three different classifiers, find accuracies and predicted values