from scipy.misc import imresize

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def draw_character(character):
  imgplot = plt.imshow(character, cmap = cm.Greys_r)
  plt.show()
