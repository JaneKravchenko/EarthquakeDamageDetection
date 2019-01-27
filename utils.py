from imageproc import Image
import numpy as np
from tqdm import tqdm
import os

def get_training_set(image_path, mask_path):
    ids = os.listdir(image_path)
    X = []
    Y = []
  
    for i in tqdm(ids):
        img = Image('{}/{}'.format(image_path, i), 'image')
        #img.normalize()
        img.set_band_type('last')
        msk = Image('{}/{}'.format(mask_path, i), 'mask')
        msk.set_band_type('last')
        X.append(img.get_array())
        Y.append(msk.get_array())

    X = np.array(X)
    X = X/X.max()  
    return X, np.array(Y)