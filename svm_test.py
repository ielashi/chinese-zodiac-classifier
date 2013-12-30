from preprocess import *
from visual import *
from sklearn import cross_validation, svm

X, y = load_preprocessed_dataset('data/rescaled28x28.dat')
X_test, _ = preprocess('data/test1.dat')

clf = svm.SVC(
    C=2., kernel='poly', degree=9, gamma=1./512., coef0=1).fit(X, y)

yhat = clf.predict(X_test)

# Output the predicted classes, one per line
for i in yhat:
  print i
