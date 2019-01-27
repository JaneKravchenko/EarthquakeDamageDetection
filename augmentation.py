import datetime
import os
import random
from copy import deepcopy
import scipy.ndimage
import cv2
import keras.backend as kb
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imageio
from imgaug import augmenters as iaa
from tqdm import tqdm
from imageproc import ImageProcessing

class SegmentAugmentation(object):
    def __init__(self):
        self.default_param = {'affine_rotate': False,
                              'affine_scale': False,
                              'affine_shear': False,
                              'affine_flip_vertical': False,
                              'affine_flip_horizontal': False,
                              'affine_probability': 0,
                              'non_affine_saturation': False,
                              'non_affine_brightness': False,
                              'non_affine_contrast': False,
                              'non_affine_shrpen': False,
                              'non_affine_grayscale': False,
                              'non_affine_emboss': False,
                              'non_affine_probability': 0,
                              'noise_blur': False,
                              'noise_noise': False,
                              'noise_dropout': False,
                              'noise_salt_and_pepper': False,
                              'noise_frequency': False,
                              'noise_probability': 0
                              }
        self.sequenceType = None
        self.mixed = None
        self.image_percent = None
        self.seq = None
        self.ids = [0]
        self.classes_count = 1

    def concatenateImage(self, image, mask):
        if image.ndim == 2:
            image = np.expand_dims(image, axis=0)
        elif image.ndim == 3:
            if min(image.shape) == image.shape[-1]:
                image = np.array(cv2.split(image))
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)
        elif mask.ndim == 3:
            self.classes_count = min(mask.shape)
        return cv2.merge(np.concatenate((image, mask), axis=0))

    def readImageGenerator(self, rastr_dir, mask_dir, get_random):
        ids_rastr = [i.split('.')[0] for i in os.listdir(rastr_dir)]
        ids_mask = [i.split('.')[0] for i in os.listdir(mask_dir)]
        formats = os.listdir(mask_dir)[0].split('.')[-1]
        ids = [i for i in ids_rastr if i in ids_mask]
        self.ids = ids
        masks_files = [os.path.join(mask_dir, i) for i in ids]
        rastr_files = [os.path.join(rastr_dir, i) for i in ids]
        if get_random:
            try:
                count = random.choice(range(len(ids)))
                rastr = ImageProcessing.read("{}.{}".format(rastr_files[count], formats))
                mask = ImageProcessing.read("{}.{}".format(masks_files[count], formats))
                concatenated = self.concatenateImage(rastr, mask)
                rastr = None
                mask = None
                return concatenated
            except Exception as e:
                print(e)

        else:
            for count, _ in enumerate(ids):
                try:
                    rastr = ImageProcessing.read(
                        "{}.{}".format(rastr_files[count], formats))
                    mask = ImageProcessing.read(
                        "{}.{}".format(masks_files[count], formats))
                    concatenated = self.concatenateImage(rastr, mask)
                    rastr = None
                    mask = None
                    yield concatenated
                except Exception as e:
                    print(e)

    def getRandomImage(self, rastr_dir, mask_dir):
        ids_rastr = [i.split('.')[0] for i in os.listdir(rastr_dir)]
        ids_mask = [i.split('.')[0] for i in os.listdir(mask_dir)]
        formats = os.listdir(mask_dir)[0].split('.')[-1]
        ids = [i for i in ids_rastr if i in ids_mask]
        masks_files = [os.path.join(mask_dir, i) for i in ids]
        rastr_files = [os.path.join(rastr_dir, i) for i in ids]
        count = random.choice(range(len(ids)))
        rastr = ImageProcessing.read("{}.{}".format(rastr_files[count], formats))
        mask = ImageProcessing.read("{}.{}".format(masks_files[count], formats))
        concatenated = self.concatenateImage(rastr, mask)
        rastr = None
        mask = None
        return concatenated

    def setSeqParam(self, sequenceType, mixed, image_percent, kwargs={}):
        """
        param: sequenceType: can be a one pf this ['one_to_one', 'random_5', 'random_10']
        """
        default_keys = list(self.default_param.keys())
        for k in default_keys:
            if k in list(kwargs.keys()):
                self.default_param[k] = kwargs[k]

    def createSequence(self):

        seq_affine = []
        seq_non_affine = []
        seq_noise = []

        def sometimes_affine(aug): return iaa.Sometimes(
            self.default_param['affine_probability']/100, aug)

        def sometimes_non_affine(aug): return iaa.Sometimes(
            self.default_param['non_affine_probability']/100, aug)

        def sometimes_noise(aug): return iaa.Sometimes(
            self.default_param['noise_probability']/100, aug)

        affine = {k: self.default_param[k] for k in self.default_param.keys() if (
            ('affine' in k) and (not 'non' in k) and self.default_param[k])}
        if affine != {}:
            affine_seq = {}
            for a in affine:
                if a == 'affine_flip_horizontal':
                    seq_affine.append(sometimes_affine(
                        iaa.Flipud(affine[a]/100)))
                elif a == 'affine_flip_vertical':
                    seq_affine.append(sometimes_affine(
                        iaa.Fliplr(affine[a]/100)))
                elif a == 'affine_rotate':
                    affine_seq['rotate'] = (-affine[a], affine[a])
                elif a == 'affine_scale':
                    affine_seq['scale'] = {
                        "x": (1, 1+affine[a]/100), "y": (1, 1+affine[a]/100)}
                elif a == 'affine_shear':
                    affine_seq['shear'] = (-affine[a], affine[a])
            if affine_seq != {}:
                affine_seq['mode'] = 'reflect'
                seq_affine.append(sometimes_affine(iaa.Affine(**affine_seq)))

        non_affine = {k: self.default_param[k] for k in self.default_param.keys() if (
            ('non_affine' in k) and self.default_param[k])}
        if non_affine != {}:
            for a in non_affine:
                if a == 'non_affine_brightness':
                    seq_non_affine.append(sometimes_non_affine(iaa.Multiply(
                        (1-non_affine[a]/100, 1+non_affine[a]/100), per_channel=True)))

                elif a == 'non_affine_contrast':
                    seq_non_affine.append(sometimes_non_affine(
                        iaa.ContrastNormalization((1-non_affine[a]/100, 1+non_affine[a]/100))))
                elif a == 'non_affine_emboss':
                    seq_non_affine.append(sometimes_non_affine(iaa.Emboss(
                        alpha=(non_affine[a]/200, non_affine[a]/100), strength=(0, non_affine[a]/50))))
                elif a == 'non_affine_grayscale':
                    seq_non_affine.append(sometimes_non_affine(
                        iaa.Grayscale(alpha=(non_affine[a]/200, non_affine[a]/100))))
                elif a == 'non_affine_saturation':
                    seq_non_affine.append(sometimes_non_affine(
                        iaa.AddToHueAndSaturation((-non_affine[a], non_affine[a]))))
                elif a == 'non_affine_shrpen':
                    seq_non_affine.append(sometimes_non_affine(iaa.Sharpen(alpha=(
                        non_affine[a]/200, non_affine[a]/100), lightness=(1-non_affine[a]/100, 1+non_affine[a]/100))))

        noise = {k: self.default_param[k] for k in self.default_param.keys() if (
            ('noise' in k) and self.default_param[k])}
        if noise != {}:
            for a in noise:
                if a == 'noise_blur':
                    seq_noise.append(sometimes_noise(
                        iaa.GaussianBlur((noise[a]/2, noise[a]))))
                elif a == 'noise_dropout':
                    seq_noise.append(sometimes_noise(
                        iaa.Dropout((noise[a]/200, noise[a]/100))))
                elif a == 'noise_frequency':
                    seq_noise.append(sometimes_noise(
                        iaa.FrequencyNoiseAlpha(exponent=(-noise[a], noise[a]))))
                elif a == 'noise_noise':
                    seq_noise.append(sometimes_noise(
                        iaa.AdditiveGaussianNoise((255*noise[a]/200, 255*noise[a]/100))))
                elif a == 'noise_salt_and_pepper':
                    seq_noise.append(sometimes_noise(
                        iaa.SaltAndPepper(p=(noise[a]/200, noise[a]/100))))

        return {'affine': seq_affine, 'non_affine': seq_non_affine, 'noise': seq_noise}

    def normalize_mask(self, mask):

        def b(band):
            band[band > ((band.max()-band.min())/2)] = 255
            band[band <= ((band.max()-band.min())/2)] = 0
            return band

        return cv2.merge([b(i) for i in cv2.split(mask)])

    def augment(self, concatenated_image, seq):
        if self.mixed == True:
            seq = {i: iaa.Sequential(seq[i]) for i in seq if seq[i] != []}
        else:
            seq = {i: [s for s in seq[i]] for i in seq if seq[i] != []}

        if random.choice(range(100)) < self.image_percent:

            if self.sequenceType == 'one_to_one':
                count = 1
            elif self.sequenceType == 'random_5':
                count = 5
            elif self.sequenceType == 'random_10':
                count = 10

            images = []
            for _ in range(count):
                if not self.mixed:
                    if 'affine' in seq.keys():
                        affine = random.choice(seq['affine'])
                        concatenated_image = affine.augment_image(
                            concatenated_image)
                    mask = concatenated_image[:, :, -self.classes_count:]
                    image = concatenated_image[:, :, :-self.classes_count]
                    effects = []
                    for i in list(seq.keys()):
                        if i != 'affine':
                            effects.append(random.choice(seq[i]))
                    if effects != []:
                        effect = random.choice(effects)
                        image = effect.augment_image(image)
                else:
                    if 'affine' in seq.keys():
                        concatenated_image = seq['affine'].augment_image(
                            concatenated_image)
                    mask = concatenated_image[:, :, -self.classes_count:]
                    image = concatenated_image[:, :, :-self.classes_count]
                    for i in list(seq.keys()):
                        if i != 'affine':
                            image = seq[i].augment_image(image)
                image -= image.min()
                image = image/image.max()
                if mask.max() == 1:
                    mask = mask.astype('int8')
                images.append((image, mask))
            return images


def augment_from_path(image_path,
            mask_path,
            param =   {'affine_rotate': 45,
                              'affine_scale': 15,'affine_shear': 15,
                              'affine_flip_vertical': True,
                              'affine_flip_horizontal': True,
                              'affine_probability': 100,
                              'non_affine_saturation': 50,
                              'non_affine_brightness': 50,
                              'non_affine_contrast': 50,
                              'non_affine_shrpen': 50,
                              'non_affine_grayscale': 50,
                              'non_affine_emboss': 50,
                              'non_affine_probability': 100,
                              'noise_blur': 2,
                              'noise_noise': 10,
                              'noise_dropout': 10,
                              'noise_salt_and_pepper': 10,
                              'noise_frequency': False,
                              'noise_probability': 100},
                              sequenceType = 'random_5',
                              mixed = True,
                              image_percent = 100):
    


    def convert_mask_ndim(mask):
        if mask.ndim == 3 and min(mask.shape) == 1:
            if min(mask.shape) == mask.shape[0]:
                mask = mask[0, :, :]
            elif min(mask.shape) == mask.shape[-1]:
                mask = mask[:, :, 0]
        return mask

    if min(imageio.volread(os.path.join(image_path, os.listdir(image_path)[0])).shape)>=4:
        param['non_affine_saturation'] = False
        param['non_affine_brightness'] = False
        param['non_affine_grayscale'] = False

    sa_process = SegmentAugmentation()
    concatenated_images = sa_process.readImageGenerator(
            image_path, mask_path, False)
    sa_process.setSeqParam(sequenceType, mixed,image_percent, param)
    formats_mask = os.listdir(mask_path)[0].split('.')[-1]
    formats_image = os.listdir(image_path)[0].split('.')[-1]
    sa_process.sequenceType = sequenceType
    sa_process.mixed = mixed
    sa_process.image_percent = image_percent
    seq = sa_process.createSequence()
    for count, ci in tqdm(enumerate(concatenated_images)):

        augmented_images = sa_process.augment(ci, seq)
        if augmented_images != []:
            for aug_num, (image, mask) in enumerate(augmented_images):
                basename = sa_process.ids[count]
                toSaveMask = os.path.join(mask_path, "{}_augment_{}.{}".format(basename,aug_num,formats_mask))
                toSaveImage = os.path.join(image_path, "{}_augment_{}.{}".format(basename,aug_num,formats_image))
                imageio.imwrite(toSaveImage, (image*255).astype('uint8'))
                mask = convert_mask_ndim(np.array(cv2.split(mask)))
                if mask.ndim == 2:
                    imageio.imwrite(toSaveMask, mask)
                else:   
                    imageio.volwrite(toSaveMask, mask)
