import numpy as np 
import sklearn
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.multiclass import OneVsOneClassifier
import scipy 
from scipy.sparse import csr_matrix, linalg

# gabor filter equation
def G(x,y,lamda,theta,psi,sigma,gamma): 
	xprime = x*np.sin(theta) + y*np.sin(theta)
	yprime = -x*np.sin(theta) + y*np.cos(theta)
	return np.exp(-(xprime**2 + (gamma*yprime)**2)/(2*sigma**2))*np.exp(complex(0,1)*(2*np.pi*xprime/lamda + psi))

# define constants
numex = 6144 # number of examples = 6144 
numpixels = 13000 # number of pixels we are using
K = 5 # folds for cross-validation
psi = 0.
lamda = 8.
sigma = np.pi
gamma = 1.
threshold = .05 # threshold below which pixel values are considered to be noise 

# initialize vectors to hold data 
X = np.zeros((numex, numpixels))
Y = []

# i got the mungies real bad 
with open('data/ml2013final_train.dat') as fin: 
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
					X[index][key] = 1.
			index += 1
X = csr_matrix(X[:index])
Y = np.array(Y)

# classify
clf = SVC(C=2., kernel='poly', degree=9, gamma=1./512., coef0=1) # classifier
preds = clf.fit(X, Y).predict(X)
errors = [preds[j] == Y[j] for j in xrange(len(Y))].count(False)
E_in = float(errors)/float(len(Y))
print "E_in", E_in

# classify with cross-validation 
kf = KFold(len(Y), n_folds=K)
E_CV_array = [] 
for train_index, test_index in kf: 
	X_train, X_test = X[train_index], X[test_index] 
	Y_train, Y_test = Y[train_index], Y[test_index]
	clf.fit(X_train, Y_train)
	preds = clf.predict(X_test)
	this_iter_error = float([preds[i] == Y_test[i] for i in xrange(len(Y_test))].count(False))/float(len(Y_test))
	E_CV_array.append(this_iter_error)
E_CV = 1./float(K)*float(sum(E_CV_array))
print "E_CV:", E_CV