import imageio
import cv2
import numpy as np
from PIL import Image
from scipy import misc
import scipy
import os


class Image(object):
    def __init__(self, path, is_mask):
        self.path = path
        self.is_mask = is_mask
        self.array = None
        self.band_count = None

        # One of tree type 'last' or 'first' ot '2d'
        self.band_type = None
        self._read()

    def _read(self):
        image_format = self.get_format()
        read_func = {'tif':self._read_tif, 'png': self._read_png, 'jpeg':self._read_jpeg}
        self.array = read_func[image_format]()
        self.band_type = self.get_band_type()
        
    def _read_tif(self):
        img = imageio.volread(self.path)
        return img

    def _read_png(self):
        img = imageio.imread(self.path)
        return img

    def _read_jpeg(self):
        img = imageio.imread(self.path)
        return img

    def get_array(self):
        return self.array

    def set_band_type(self, to_type):
        if self.band_type!=to_type:
            if to_type == 'last' and self.band_type == '2d':
                self.array = np.expand_dims(self.array, axis = 2)
                self.band_type == 'last'
            elif to_type == 'last' and self.band_type!='2d':
                self.array = np.array(cv2.merge(self.array))
                self.band_type == 'last'
            elif to_type == 'first' and self.band_type == '2d':
                self.array = np.expand_dims(self.array, axis = 0)
                self.band_type == 'first'
            elif to_type == 'first' and self.band_type!='2d':
                self.array = np.array(cv2.split(self.array))
                self.band_type == 'first'
            elif to_type == '2d' and self.band_cout == 1:
                if self.band_type == 'first':
                    self.array = self.array[0]
                    self.band_type == '2d'
                elif self.band_type == 'last':
                    self.array = self.array[:, :, -1]
                    self.band_type == '2d'
                

    def get_band_type(self, array = None):
        band_count = self.get_band_count(array)
        if band_count!=1:
            if type(array)==type(None):
                if self.array.shape[-1] == band_count:
                    return 'last'
                elif self.array.shape[0] == band_count:
                    return 'first'
            elif type(array)==np.ndarray:
                if array.shape[-1] == band_count:
                    return 'last'
                elif array.shape[0] == band_count:
                    return 'first'
        else:
            return '2d'

    def get_band_count(self, array = None):
        if type(array)==type(None):
            if self.array.ndim==3:
                return min(self.array.shape)
            else:
                return 1
        elif type(array) == np.ndarray:
            if array.ndim == 3:
                return min(array.shape)
            else:
                return 1
        else:
            raise ValueError('Array is requires')

    def get_format(self):
        return self.path.split('.')[-1]

    def _normalize_band(self, band):
        max_ = np.max(band)
        min_ = np.min(band)
        alpha = max_-min_
        if alpha!=0:
            return (band-min_)/alpha
        else:
            return band


    def normalize(self):
        previous_band_type = self.band_type
        if self.band_type == 'last' or self.band_type == '2d':
            self.set_band_type('first')
        self.array = np.array(cv2.merge([self._normalize_band(band) for band in self.array]))
        self.set_band_type(previous_band_type)


class ImageProcessing(object):
    
    @classmethod
    def stretch(self, array, dim = None):  
        if array.shape[-1]<=11:
            array = np.array(cv2.split(array))
        if not dim:
            dim = array.shape[1]
        channels= np.array(array).astype('float32')
        channels = np.array([scipy.misc.imresize(i, (dim, dim), 'lanczos') for i in channels])
        alphas = [np.max(i) - np.min(i) for i in channels]
        channels_normalize = np.array([(im-np.min(im))/alphas[count] for count, im in enumerate(channels)])
        return channels
    
    @classmethod
    def read(self, path):
        
        if path.split('.')[-1] == 'tif':
            array = imageio.volread(path)
            if array.shape[-1]<=11:
                array = np.array(cv2.split(array))
            return array
        
        elif path.split('.')[-1] == 'jpg':
            return np.array(cv2.split(np.array(Image.open(path))))

        elif path.split('.')[-1] == 'png':
            array = imageio.imread(path).astype('uint8')
            if min(array.shape)==array.shape[0]:
                return array
            elif min(array.shape) == array.shape[-1]:
                return np.array(cv2.split(array))
        
    @classmethod
    def save(self, path, array):
        if path.split('.')[-1] == 'tif': 
            if array.shape[-1]<=11:
                array = cv2.split(array)
            imageio.mimwrite(path, array)
        elif path.split('.')[-1] == 'jpg': 
            if array.shape[0]<4:
                array = cv2.merge(array)
            imageio.imwrite(path, array, 'jpeg')
     
    
    @classmethod 
    def resize(self, array, rows, cols):
        if array.shape[-1]<=11:
            array = cv2.split(array)  
            array = np.array([scipy.misc.imresize(i, (rows, cols), 'lanczos') for i in array])
            return array

        else:
            array = np.array([scipy.misc.imresize(i, (rows, cols), 'lanczos') for i in array])
            return cv2.merge(array)
               
    @classmethod
    def resizeAndSave(self, array, path, rows, cols):
        array = self.resize(array, rows, cols)
        self.save(path, array)
        
        
    @classmethod        
    def stretchAndSave(self, array, dim, path):
        array = self.stretch(array, dim)
        self.save(path, array)
        

    @classmethod            
    def getImageFromDir(self, dir, format = None):
        files = os.listdir(dir)
        if format:
            return [os.path.join(dir, i) for i in files if i.split('.')[-1] == format]
        else:
            return [os.path.join(dir, i) for i in files]


    @classmethod
    def openForTraining(self, path):
        try:
            array = self.read(path)
            return cv2.merge(array)
        except Exception as e:
            logging.error("Error in line {} {}: {}".format(e.__traceback__.tb_lineno, e.__class__.__name__, e))


    @classmethod
    def openAndResize(self, path, rows = None, cols = None):
        array = self.read(path)
        try:
            if not (rows and cols):
                shape = array.shape
                if shape[0]<=11:
                    rows = max(shape[1], shape[2])
                    cols = rows
                else:
                    rows = max(shape[0], shape[1])
                    cols = rows
            return self.resize(array, rows, cols)
        except Exception as e: 
            logging.error("Error in line {} {}: {}".format(e.__traceback__.tb_lineno, e.__class__.__name__, e))
            return np.ones(array.shape)
