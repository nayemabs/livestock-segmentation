
import numpy as np
from PIL import Image
from numba import jit
import os
import glob

@jit(nopython=True)

def index2index(img):
    print(img.shape)
    actual_area = 103.2256
    sticker_pixel_count = 0
    cow_pixel_count = 0
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x,y] == 0:
                sticker_pixel_count += 1
            if img[x,y] == 1:
                cow_pixel_count +=1

    cow_pixel_count = cow_pixel_count+sticker_pixel_count
    cow_area = cow_pixel_count*(actual_area/sticker_pixel_count)
    cow_length = 1200*(actual_area/sticker_pixel_count)
                
    print(sticker_pixel_count)
    print(cow_area)
    print(cow_length)

animal = 'cow'
type = 'Side'
in_dir = f'/home/abs/Datasets/{animal}/{type}/images/fuse_binary/'


for im in os.listdir(in_dir):
	print(im)
	img = Image.open(f"{in_dir}/{im}")
	img = np.asarray(img)
	out = index2index(img)


print('\nDone!!')