# Utility functions for pre-processing the raw dataset

from math import floor
from scipy.misc import imresize

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def load_dataset(filename):
  f = open(filename, 'r')

  dataset = []

  for line in f:
    data = line.strip().split(' ')
    classification = data[0]
    features = data[1:]
    if len(features) >= 20:
      character = np.array([0.] * 12810)

      for feature in features:
        (i, feature_value) = feature.split(':')
        character[int(i) - 1] = float(feature_value)

      character = character.reshape(122, 105)

      dataset.append((character, classification))

  return dataset


def load_test_dataset(filename):
  f = open(filename, 'r')

  dataset = []

  for line in f:
    data = line.strip().split(' ')
    classification = data[0]
    features = data[1:]
    character = np.array([0.] * 12810)

    for feature in features:
      (i, feature_value) = feature.split(':')
      character[int(i) - 1] = float(feature_value)

    character = character.reshape(122, 105)

    dataset.append((character, classification))

  return dataset

def load_preprocessed_dataset(filename):
  f = open(filename, 'r')
  dataset = []
  for line in f:
    classification, character = line.strip().split(' ')
    character = np.array([float(i) for i in character.split(',')])
    dataset.append((character, int(classification)))

  X = np.array([e[0] for e in dataset])
  y = np.array([e[1] for e in dataset])

  return X, y


def crop_bounding_box(dataset):
  new_dataset = []
  for character, _ in dataset:
    B = np.argwhere(character)
    try:
      (ystart, xstart), (ystop, xstop) = B.min(0), B.max(0) + 1
      character = character[ystart:ystop, xstart:xstop]
      new_dataset.append((character, _))
    except:
      # something is wrong with the image (empty image?)
      # add image as is
      new_dataset.append((character, _))

  return new_dataset


def make_binary(dataset, threshold=10):
  new_dataset = []

  for character, _ in dataset:
    character[character < threshold] = 0
    character[character >= threshold] = 1

    new_dataset.append((character, _))

  return new_dataset


def resize_images(dataset, width, height):
  return [(imresize(ch, (width, height)) / 256.0, cl) for ch, cl in dataset]


def output_dataset(dataset):
  for character, class_ in dataset:
    print class_, ','.join(['%f' % num for num in character.ravel()])


def preprocess(filename):
  dataset = load_dataset(filename)
  dataset = crop_bounding_box(dataset)
  dataset = resize_images(dataset, 28, 28)

  X = [e[0].ravel() for e in dataset]
  y = [e[1] for e in dataset]

  return X, y

def test_preprocess(filename):
  dataset = load_test_dataset(filename)
  dataset = crop_bounding_box(dataset)
  dataset = resize_images(dataset, 28, 28)

  X = [e[0].ravel() for e in dataset]
  y = [e[1] for e in dataset]

  return X, y
