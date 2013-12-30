from preprocess import *
from visual import *


X, y = preprocess('data/ml2013final_train.dat')

dataset = zip(X, y)

output_dataset(dataset)
