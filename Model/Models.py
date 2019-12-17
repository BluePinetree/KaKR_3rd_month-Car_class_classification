import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
# from efficientnet.keras import EfficientNetB3
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPool2D, Dropout, MaxPooling2D

def build_model(model_name=None, include_top=False, input_shape=(256,256,3), fine_tuning=True, layer_to_freeze=None, load_pretrained : str = None, summary=False):
    pmodel_name = model_name.strip().lower()
    print(pmodel_name)

    if pmodel_name == 'resnet50': base_model = ResNet50(include_top=include_top, input_shape=input_shape)
    elif pmodel_name == 'inception_v3' : base_model = InceptionV3(include_top=include_top, input_shape=input_shape)
    elif pmodel_name == 'xception' : base_model = Xception(include_top=include_top, input_shape=input_shape)
    # elif pmodel_name == 'efficient_net' : base_model = EfficientNetB3(include_top=include_top, input_shape=input_shape)
    else : raise ValueError

    if fine_tuning:
        # Freese layers
        assert layer_to_freeze != None, 'You must define layer\'s name to freese.'
        fr_layer_name = layer_to_freeze
        set_trainable = False

        for layer in base_model.layers:
            if not layer.name == fr_layer_name:
                set_trainable = True

            layer.trainable = set_trainable

    # change last layers
    last_1dconv_1 = Conv2D(1024, 1, activation='relu', kernel_initializer='he_normal')(base_model.output)
    last_pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(last_1dconv_1)
    global_avg_pool = GlobalAveragePooling2D()(last_pool_1)
    last_Dense_1 = Dense(512, activation='relu', kernel_initializer='he_normal')(global_avg_pool)
    dropout_1 = Dropout(rate=0.5)(last_Dense_1)
    last_Dense_2 = Dense(196, activation='softmax')(dropout_1)

    # compile
    model = Model(base_model.input, last_Dense_2)

    # summary
    if summary:
        model.summary()

    # load pretrained weights
    if load_pretrained:
        model.load_weights(load_pretrained)

    return model
