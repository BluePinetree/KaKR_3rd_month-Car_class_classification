import gc
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K

from sklearn.metrics import f1_score
from Models import *

def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# Define steps per epoch
def get_steps(num_samples, batch_size):
    if (num_samples // batch_size) > 0:
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size

# https://www.kaggle.com/seriousran/cutout-augmentation-on-keras-efficientnet
def get_random_erazer(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def erazer(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p :
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s/r))
            h = int(np.sqrt(s*r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return erazer


# For prediction
def predict_class(model, num_samples, batch_size, train_generator, test_generator, output_name, DATA_PATH = './input', OUTPUT_PATH = './output'):

    # Prediction
    prediction = model.predict_generator(
        generator=test_generator,
        steps=get_steps(num_samples, batch_size),
        verbose=1
    )

    predicted_indices = np.argmax(prediction, axis=1)
    # print(predicted_indices[:10])
    labels = (train_generator.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_indices]
    # print(predictions[:10])
    # Load submission form
    submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
    submission['class'] = predictions
    submission.to_csv(os.path.join(OUTPUT_PATH, '{}.csv'.format(output_name)), index=False)

def predict_class_ensemble(models, weights, num_samples, batch_size, train_generator, test_generator, DATA_PATH = './input', OUTPUT_PATH = './output'):
    predictions = []


    if not models == None:
        num_models = len(models)

        if not weights == None:
            num_weights = len(weights)
            for model_name in models:
                for i, weight_name in enumerate(weights):
                    print('=== predict {0} model - {1}\'s split ==='.format(model_name, i))
                    model = build_model(model_name=model_name, input_shape=(299,299,3), fine_tuning=False, summary=False)
                    model.load_weights(os.path.join('./weights', weight_name))

                    test_generator.reset()

                    prediction = model.predict_generator(
                        generator=test_generator,
                        steps=get_steps(num_samples, batch_size),
                        verbose=1
                    )

                    predictions.append(prediction)

        else:
            for model_name in models:
                print('=== predict {} model ==='.format(model_name))
                model = build_model(model_name=model_name, input_shape=(299, 299, 3), fine_tuning=False, summary=False)
                model.load_weights(os.path.join('./weights', '{}_model_50_epochs.h5'.format(model_name)))

                # 한번 예측 후에는 반드시 리셋 필수!
                test_generator.reset()

                prediction = model.predict_generator(
                    generator=test_generator,
                    steps=get_steps(num_samples, batch_size),
                    verbose=1
                )

                print(np.argmax(prediction, axis=-1)[:10])

                predictions.append(prediction)

        print('Complete!')
        predictions = np.array(predictions)
        print(predictions.shape)

        predictions = np.mean(predictions, axis=0)

        # 제출 형식으로 변환
        predict_indices = np.argmax(predictions, axis=-1)
        labels = (train_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        prediction_ensemble = [labels[k] for k in predict_indices]

        # 제출
        output_name = 'ensemble(' + ','.join(models) + ')_{}_splits.csv'.format(num_weights)
        submission = pd.read_csv(os.path.join(DATA_PATH, 'sample_submission.csv'))
        submission['class'] = prediction_ensemble
        submission.to_csv(os.path.join(OUTPUT_PATH, output_name), index=False)