from preprocess import *
from visual import *


dataset = load_dataset()
dataset = crop_bounding_box(dataset)
dataset = resize_images(dataset, 20, 20)
dataset = make_binary(dataset)

#draw_character(dataset[0][0])
#draw_character(dataset[1][0])
#draw_character(dataset[2][0])
#draw_character(dataset[3][0])
#draw_character(dataset[4][0])
output_dataset(dataset)
