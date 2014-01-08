"""
This is a support vector machine classifier for handwritten Chinese text.
"""

import numpy as np 
import math
import sklearn
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
import scipy 
from scipy.sparse import csr_matrix, linalg
import os

from preprocess import *

# define constants
K = 5 # number of folds for cross-validation

# load data 
print "Loading data..."
dataset = load_dataset('data/ml2013final_train.dat')
dataset = crop_bounding_box(dataset)
dataset = resize_images(dataset, 40, 40)
# initialize vectors to hold data 
X = []
Y = []
for pair in dataset: 
	(pixels, label) = pair 
	X.append(pixels.flatten())
	Y.append(label)
X = np.array(make_binary(X))
Y = np.array(Y)
print X 
print Y
print "Done."

# classify
print "Classifying..."
clf = SVC(C=2., kernel='poly', degree=11, gamma=1./2048., coef0=1) 
preds = clf.fit(X, Y).predict(X)
print "Done."
print "Computing E_in..."
errors = [preds[j] == Y[j] for j in xrange(len(Y))].count(False)
E_in = float(errors)/float(len(Y))
print "E_in", E_in
print "Done."

# predict for test set 
# load test data 
print "Loading test data..."
dataset = load_dataset('data/ml2013final_test1.nolabel.dat')
dataset = crop_bounding_box(dataset)
dataset = resize_images(dataset, 40, 40)
test_X = []
test_Y = []
for pair in dataset: 
	(pixels, label) = pair 
	test_X.append(pixels.flatten())
test_X = np.array(make_binary(test_X))
print "Done."
print "Making test predictions..."
test_preds = clf.predict(test_X)
print test_preds
# write to file
with open('%s_preds.txt' % str(os.path.basename(__file__)).split('.')[0], 'w') as fout: 
	for y in test_preds: 
		fout.write(str(y)+'\n')
print "Done."

# cross-validate
print "Cross-validating..."
kf = KFold(len(Y), n_folds=K)
E_CV_array = [] 
for train_index, test_index in kf: 
	X_train, X_test = X[train_index], X[test_index] 
	Y_train, Y_test = Y[train_index], Y[test_index]
	X_train = make_binary(X_train)
	X_test = make_binary(X_test)
	clf.fit(X_train, Y_train)
	preds = clf.predict(X_test)
	this_iter_error = float([preds[i] == Y_test[i] for i in xrange(len(Y_test))].count(False))/float(len(Y_test))
	print "this_iter_error", this_iter_error
	E_CV_array.append(this_iter_error)
E_CV = 1./float(K)*float(sum(E_CV_array))
print "E_CV:", E_CV
print "Done."