from imageproc import imageproc
import numpy as np
from tqdm import tqdm

def get_training_set(image_path, mask_path):
  ids = os.listdir(image_path)
  X = []
  Y = []
  
  for i in tqdm(ids):
    img = Image(f'{image_path}/{i}', 'image')
    img.normalize()
    img.set_band_type('last')
    msk = Image(f'{mask_path}/{i}', 'mask')
    msk.set_band_type('last')
    X.append(img.get_array())
    Y.append(msk.get_array())
  
  return np.array(X), np.array(Y)