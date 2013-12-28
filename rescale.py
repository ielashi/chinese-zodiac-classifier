from preprocess import *
from visual import *


dataset = load_dataset()
dataset = resize_images(dataset, 20, 20)
output_dataset(dataset)
