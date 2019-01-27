from keras.layers import *
from keras.models import Model
from keras.optimizers import *
from segmentation_models import *
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
#from keras.applications.densenet import DenseNet
#from keras.applications.nasnet import NASNet


def get_model(model_name, model_type = 'convnet', classes_count = 1, optimizer = 'Adam', metrics = ['acc'],  weights = 'imagenet', input_shape = (224, 224, 3), savepath = None):
    keras_pretrained = {'vgg16': VGG16,
                        'vgg19': VGG19,
                        'xception': Xception,
                        'resnet50': ResNet50,
                        'inception_v3':InceptionV3,
                        'inception_resnet_v2':InceptionResNetV2,
                        'mobilenet':MobileNet}

    if model_type == 'convnet':
        input_model = keras_pretrained[model_name](include_top = False, weights = weights, input_shape = input_shape)
        inputs = input_model.input
        input_model_output = input_model.output
        flatten = Flatten()(input_model_output)
        dense1 = Dense(256, activation = 'relu')(flatten)
        dropout = Dropout(0.85)(dense1)

        if classes_count == 1:
            last_activation = 'sigmoid'
        else:
            last_activation = 'softmax'

        dense2 = Dense(classes_count, activation = last_activation)(dropout)

        model = Model(inputs = inputs, outputs = dense2)

        if classes_count==1:
            model.compile(loss = 'binary_crossentropy', metrics = metrics, optimizer = optimizer)
        else:
            model.compile(loss = 'categorical_crossentropy', metrics = metrics, optimizer = optimizer)
            
        if savepath:
            model.save(savepath)
        return model

    elif model_type == 'unet':
        model = Unet(backbone_name=model_name, encoder_weights=weights, input_shape = input_shape)
        if classes_count==1:
            model.compile(loss = 'binary_crossentropy', metrics = metrics, optimizer = optimizer)
        else:
            model.compile(loss = 'categorical_crossentropy', metrics = metrics, optimizer = optimizer)

        if savepath:
            model.save(savepath)

        return model
