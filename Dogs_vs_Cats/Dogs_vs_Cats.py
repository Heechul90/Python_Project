### Dogs vs Cats
# Kaggle Dataset의 일부를 이용한 개, 고양이 구분
# Dog Image: 1,111개, Cat Image: 1,111개, 총 2,222개
# 출처: pontoregende GitHub

# 함수 준비하기
from keras.preprocessing import image
from glob import glob
import cv2, os, random
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# Data dimensions and paths
img_width = 150
img_height = 150
TRAIN_DIR = '../dataset/dogs-vs-cats/train/'
TEST_DIR = '../dataset/dogs-vs-cats/test1/'
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]