import imageio
import cv2
import numpy as np



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

