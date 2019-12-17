import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import PIL

# https://www.kaggle.com/tmheo74/3rd-ml-month-car-image-cropping-updated-7-10

# 데이터 경로
TRAIN_IMG_PATH = os.path.join(os.getcwd(), 'input/train')
TEST_IMG_PATH = os.path.join(os.getcwd(), 'input/test')
DATA_PATH = os.path.join(os.getcwd(), 'input')

def crop_boxing_img(img, pos, margin=16):
    width, height = img.size
    x1 = max(0, pos[0] - margin)
    y1 = max(0, pos[1] - margin)
    x2 = min(width, pos[2] + margin)
    y2 = min(height, pos[3] + margin)

    cropped_img = img.crop((x1,y1,x2,y2))
    # plt.imshow(cropped_img)
    # plt.show()
    return cropped_img

def main():
    df_train = pd.read_csv(os.path.join(DATA_PATH, 'train.csv'))
    df_test = pd.read_csv(os.path.join(DATA_PATH, 'test.csv'))

    if not os.path.exists('./input/train_crop'):
        os.mkdir('./input/train_crop')

    if not os.path.exists('./input/test_crop'):
        os.mkdir('./input/test_crop')

    # 훈련 이미지 자르기
    for i, img_name in tqdm(enumerate(df_train['img_file'])):
        img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, img_name))
        pos = df_train.iloc[i][['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)
        cropped_img = crop_boxing_img(img, pos)
        cropped_img.save(os.path.join(DATA_PATH, 'train_crop/'+img_name))

    # 시험 이미지 자르기
    for i, img_name in tqdm(enumerate(df_test['img_file'])):
        img = PIL.Image.open(os.path.join(TEST_IMG_PATH, img_name))
        pos = df_test.iloc[i][['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)
        cropped_img = crop_boxing_img(img, pos)
        cropped_img.save(os.path.join(DATA_PATH, 'test_crop/' + img_name))



if __name__ == '__main__':
    main()