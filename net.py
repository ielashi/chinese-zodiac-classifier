import numpy as np 
import sklearn
from sklearn.cross_validation import KFold
import scipy 
from scipy.sparse import csr_matrix, linalg
import pybrain
from pybrain.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, SoftmaxLayer, FullConnection
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError

# define constants
numex = 6144 # number of examples = 6144 
K = 5 # folds for cross-validation
threshold = .05 # threshold below which pixel values are considered to be noise 
numpixels = 13000 # number of pixels per example 
numlayers = 6 # number of hidden layers to use in neural net 
numclasses = 12 # number of classes for classification 

# initialize vectors to hold data 
X = np.zeros((numex, numpixels))
Y = []

# i got the mungies real bad 
with open('ml2013final_train.dat') as fin: 
	index = 0 # index 
	for line in fin.readlines()[:numex]: 
		line = line.strip().split()
		if len(line) < 20: 
			pass
		else: 
			Y.append(int(line[0])) # append class label to the Y vector 
			for pixel in line[1:]: 
				key, value = pixel.strip().split(':')
				key, value = int(key), float(value)
				if value >= threshold:
					X[index][key] = value
			index += 1
X = np.array(X[:index]).reshape(index, numpixels)
Y = np.array(Y).reshape(len(Y), 1)
data = np.append(X, Y, axis=1)

# build the net 
# input dimension of number of pixels and output dimension of number of classes
# because it is best to use one output neuron per class 
net = buildNetwork(numpixels, numlayers, numclasses, outclass=SoftmaxLayer)

# train 
trainer = BackpropTrainer(net, dataset=data, momentum=0.1, weightdecay=.01)
for i in xrange(20): 
	print i
	trainer.trainEpochs(5)
	# calculate in-sample error 
	E_in = percentError(trainer.testOnClassData(dataset=X), Y)
print "E_in", E_in

"""
# cross-validation 
kf = KFold(len(Y), n_folds=K)
E_CV_array = [] 
for train_index, test_index in kf: 
	X_train, X_test = X[train_index], X[test_index] 
	Y_train, Y_test = Y[train_index], Y[test_index]
	this_iter_error = percentError(trainer.testOnClassData(dataset=X_test), Y_test)
	E_CV_array.append(this_iter_error)
E_CV = 1./float(K)*float(sum(E_CV_array))
print "E_CV:", E_CV
"""