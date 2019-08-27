# -*- coding: utf-8 -*-

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint,EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2, os, random
import time


# CIFAR10 데이터 로드
(X_train, y_train0), (X_test, y_test0) = cifar10.load_data()
print(X_train.shape, X_train.dtype)
print(y_train0.shape, y_train0.dtype)
print(X_test.shape, X_test.dtype)
print(y_test0.shape, y_test0.dtype)

# 데이터 확인
plt.subplot(141)
plt.imshow(X_train[0], interpolation = "bicubic")
plt.grid(False)
plt.subplot(142)
plt.imshow(X_train[4], interpolation = "bicubic")
plt.grid(False)
plt.subplot(143)
plt.imshow(X_train[8], interpolation = "bicubic")
plt.grid(False)
plt.subplot(144)
plt.imshow(X_train[12], interpolation = "bicubic")
plt.grid(False)
plt.show()

# 자료형 변환 및 스케일링
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

print(X_train.shape, X_train.dtype)

from keras.utils import np_utils

Y_train = np_utils.to_categorical(y_train0, 10)
Y_test = np_utils.to_categorical(y_test0, 10)
Y_train[:4]

## 모델 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (32, 32, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(Conv2D(64, (3, 3), padding = 'same', activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

## 모델 저장 폴더 설정
Model_dir = 'Project01/model3/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)

## 모델 저장 조건 설정
modelpath = 'Project01/model3/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

## 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)


# 모델 실행
%%time
hist = model.fit(X_train, Y_train,
                 epochs = 30,
                 batch_size = 200,
                 validation_data = (X_test, Y_test),
                 verbose = 2,
                 callbacks = [early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))

# 테스트셋의 오차
y_vloss = hist.history['val_loss']

# 학습셋의 오차
y_loss = hist.history['loss']

# loss 그래프 그리기
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'Trainset_loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()
