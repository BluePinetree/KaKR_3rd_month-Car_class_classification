from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import gc
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 모델링
sys.path.append(os.path.join(os.getcwd(), 'Model'))
from Models import *
from utils import *

from keras.applications.resnet50 import preprocess_input
from keras_preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras import optimizers
from keras.layers import Conv2D, Dense, Flatten, GlobalAveragePooling2D, GlobalMaxPool2D, Dropout
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


# set random seed
RANDOM_SEED = 40

# 데이터 경로 설정
DATA_PATH = './input'
OUTPUT_PATH = './output'
os.listdir(DATA_PATH)

# 이미지 경로 설정
TRAIN_IMG_PATH = os.path.join(DATA_PATH, 'train_crop')
TEST_IMG_PATH = os.path.join(DATA_PATH, 'test_crop')

# 데이터 읽어오기
df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

# 모델링을 위한 데이터 준비
df_train['class'] = df_train['class'].astype('str')

df_train = df_train[['img_file', 'class']]
df_test = df_test[['img_file']]

# Parameters
img_size = (299, 299)
epochs = 50
batch_size = 16
learning_rate = 0.0002
base_model = 'xception'
load_pretrained = None
patience = 5
n_splits=1

# Define Generator config
train_datagen = ImageDataGenerator(
    horizontal_flip = True,
    vertical_flip = False,
    zoom_range = 0.10,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range=0.5,
    brightness_range=[0.5, 1.5],
    fill_mode='nearest',
    rescale=1./255,
    preprocessing_function=get_random_erazer(v_l=0, v_h=255)
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Train, Test spllit
# X_train, X_val = train_test_split(df_train, train_size=0.95, random_state=RANDOM_SEED)
splitter = StratifiedShuffleSplit(n_splits=n_splits, train_size=0.9, test_size=0.1, random_state=RANDOM_SEED)

scores = []


# 학습
for i, (trn_idx, val_idx) in enumerate(splitter.split(df_train['img_file'], df_train['class'])):

    print('============', '{}\'th Split'.format(i), '============\n')

    X_train = df_train.iloc[trn_idx].copy()
    X_val = df_train.iloc[val_idx].copy()
    print('Train : {0} / Test : {1}'.format(X_train.shape, X_val.shape))

    train_size = len(X_train)
    val_size = len(X_val)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe = X_train,
        directory = TRAIN_IMG_PATH,
        x_col = 'img_file',
        y_col = 'class',
        target_size = img_size,
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = batch_size,
        seed=RANDOM_SEED,
        shuffle=True,
        interploation = 'bicubic'
    )

    val_generator = val_datagen.flow_from_dataframe(
        dataframe = X_val,
        directory = TRAIN_IMG_PATH,
        x_col = 'img_file',
        y_col = 'class',
        target_size = img_size,
        color_mode = 'rgb',
        class_mode = 'categorical',
        batch_size = batch_size,
        seed=RANDOM_SEED,
        shuffle=True,
        interploation = 'bicubic'
    )

    test_generator = test_datagen.flow_from_dataframe(
        dataframe = df_test,
        directory = TEST_IMG_PATH,
        x_col = 'img_file',
        y_col = None,
        target_size = img_size,
        color_mode = 'rgb',
        class_mode = None,
        batch_size = batch_size,
        shuffle=False
    )

    # model = build_Inception_v3_model(input_shape=(299,299,3), fine_tuning=False)
    model = build_model(model_name=base_model, input_shape=(299, 299, 3), fine_tuning=False, summary=False)

    # Load pretrained
    if load_pretrained is not None:
        # assert type(load_pretrained) == str
        print('Loading Pretrained networks {}'.format(load_pretrained))
        model.load_weights(os.path.join(os.getcwd(), './weights/' + load_pretrained))


    optimizer = optimizers.adam(lr=learning_rate)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', f1_m])

    if not os.path.exists('./weights'):
        os.mkdir('./weights')

    # checkpoint_filename = 'inception_v3_model_{0}_epochs.h5'.format(epochs)
    if n_splits > 1:
        checkpoint_filename = '{0}_model_{1}_epochs_split__{2}.h5'.format(base_model,epochs, i)
    else:
        checkpoint_filename = '{0}_model_{1}_epochs.h5'.format(base_model, epochs)


    # Define callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=int(patience / 2),
        verbose=1,
        mode='min',
        min_lr=0.0000001
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.000001,
        patience=patience,
        verbose=1,
        mode='min'
    )

    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join('./weights', checkpoint_filename),
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='min',
        period=5
    )

    callbacks = [reduce_lr, early_stopping, model_checkpoint]

    history = model.fit_generator(
        generator = train_generator,
        steps_per_epoch = get_steps(train_size, batch_size),
        epochs = epochs,
        callbacks = callbacks,
        validation_data = val_generator,
        validation_steps = get_steps(val_size, batch_size)
    )

    # Predict class
    # predict_class(model, df_test.shape[0], batch_size, train_generator, test_generator, checkpoint_filename[:-3])

    # Predict ensemble
    # models = ['inception_v3', 'resnet50', 'xception', 'efficient_net']
    weights = ['efficient_net_model_50_epochs_split__0.h5', 'efficient_net_model_50_epochs_split__1.h5', 'efficient_net_model_50_epochs_split__2.h5', 'efficient_net_model_50_epochs_split__3.h5', 'efficient_net_model_50_epochs_split__4.h5']
    predict_class_ensemble(['efficient_net'], weights, df_test.shape[0], batch_size, train_generator, test_generator)

    gc.collect()