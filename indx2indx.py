import numpy as np
from PIL import Image
from numba import jit
import os
import glob

@jit(nopython=True)
def im2index(im):
    assert len(im.shape) == 3
    r = im[:, :, 0].ravel()
    g = im[:, :, 1].ravel()
    b = im[:, :, 2].ravel()
    
    label = np.zeros((r.shape[0], 1), dtype=np.uint8)
    for i in range(r.shape[0]):
        if r[i] == 255 and g[i] == 240 and b[i] == 0:
            label[i] = 0
        elif r[i] == 0 and g[i] == 255 and b[i] == 0:
            label[i] = 1
        else:
            label[i] = 2
     
    indx_img = np.reshape(label, (-1, im.shape[1]))
    return indx_img


animal = 'cow'
types = ['Side','Side_2', 'Back', 'Back_2']
for type in types:
    in_dir = f'/home/abs/Datasets/{animal}/{type}/images/fuse/'
    out_dir = f"/home/abs/Datasets/{animal}/{type}/images/fuse_binary/"

    if os.path.exists(out_dir):
        print('Directory already exists')
        pass
    else:
        os.makedirs(out_dir)
        print('Directory created')


    for im in os.listdir(in_dir):
        print(im)
        img = Image.open(f"{in_dir}/{im}")
        img = np.asarray(img)
        # print(np.unique(img))
        out = im2index(img)
        Image.fromarray(out).save(f"{out_dir}/{im}")

    print('\nDone!!')