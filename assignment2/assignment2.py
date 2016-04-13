import pipeline as pp
import numpy as np

test_data = pp.data_read('data_input/cs-test.csv')
train_data = pp.data_read('data_input/cs-training.csv')
pp.data_explore(test_data, "test_set")
pp.data_explore(train_data, "train_set")
test_data = pp.mean_median_fill(test_data)
train_data = pp.mean_median_fill(train_data)
allvarslist = list(test_data)
target = 'SeriousDlqin2yrs'
allvarslist.remove(target)
clfs = ['log', 'Kneighbors', 'LSVC']
fclfs = []
accuracies = []
for clf_type in clfs:
	this_clf = pp.build_classifier(train_data, [target], allvarslist, clf_type)
	this_values = pp.clf_predict(this_clf, test_data, allvarslist)
	this_accuracy = pp.clf_accuracy(this_clf, train_data[allvarslist], train_data[target])
	print clf_type + " accuracy is " + str(this_accuracy)
	fclfs.append(this_clf)
	accuracies.append(this_accuracy)
best_classifier = fclfs[accuracies.index(max(accuracies))]
best_values = pp.clf_predict(best_classifier, test_data, allvarslist)
np.savetxt("data_output/predicted_values.txt", best_values)