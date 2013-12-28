# Utility functions for pre-processing the raw dataset

from math import floor
from scipy.misc import imresize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_dataset():
  f = open('data/ml2013final_train.dat', 'r')

  dataset = []
  for line in f:
    data = line.strip().split(' ')
    classification = data[0]
    features = data[1:]

    character = np.array([0] * 12810)

    for feature in features:
      (i, feature_value) = feature.split(':')
      character[int(i) - 1] = floor(float(feature_value) * 256)

    character = character.reshape(122, 105)

    dataset.append((character, classification))

  return dataset


def load_preprocessed_dataset(filename):
  f = open(filename, 'r')
  dataset = []
  for line in f:
    classification, character = line.strip().split(' ')
    character = np.array(character.split(','))
    dataset.append((character, int(classification)))

  X = np.array([e[0] for e in dataset])
  y = np.array([e[1] for e in dataset])

  return X, y


def resize_images(dataset, width, height):
  return [(imresize(ch, (width, height)), cl) for ch, cl in dataset]


def output_dataset(dataset):
  for character, class_ in dataset:
    print class_, ','.join(['%d' % num for num in dataset[0][0].ravel()])