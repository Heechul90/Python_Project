### Rock, Paper, Scissors

# 함수, 모듈 준비
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from glob import glob

import cv2, os, random
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

path = 'Project03/RockPaperScissors/'

## 사이즈 맞추기
ROW, COL = 96, 96

# 데이터 불러오기(Rocks)
rock_path = os.path.join(path, 'rock/*.png')
len(glob(rock_path))

Rocks = []
for rock_image in glob(rock_path):
    rock = cv2.imread(rock_image)
    rock = cv2.cvtColor(rock, cv2.COLOR_BGR2GRAY)
    rock = cv2.resize(rock, (ROW, COL))
    rock = image.img_to_array(rock)
    Rocks.append(rock)
len(Rocks)

# 데이터 불러오기(Papers)
paper_path = os.path.join(path, 'paper/*.png')
len(glob(paper_path))

Papers = []
for paper_image in glob(paper_path):
    paper = cv2.imread(paper_image)
    paper = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
    paper = cv2.resize(paper, (ROW, COL))
    paper = image.img_to_array(paper)
    Papers.append(paper)
len(Papers)

# 데이터 불러오기(Scissors)
scissor_path = os.path.join(path, 'scissors/*.png')
len(glob(scissor_path))

Scissors = []
for scissor_image in glob(scissor_path):
    scissor = cv2.imread(scissor_image)
    scissor = cv2.cvtColor(scissor, cv2.COLOR_BGR2GRAY)
    scissor = cv2.resize(scissor, (ROW, COL))
    scissor = image.img_to_array(scissor)
    Scissors.append(scissor)
len(Scissors)



## 데이터 확인하기
classes = ['rock', 'paper', 'scissors']

# Rock
plt.figure(figsize=(12,8))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = image.array_to_img(random.choice(Rocks))
    plt.imshow(img, cmap = plt.get_cmap('gray'))

    plt.axis('off')
    plt.title('It should be a {}.'.format(classes[0]))
plt.show()

# Paper
plt.figure(figsize=(12,8))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = image.array_to_img(random.choice(Papers))
    plt.imshow(img, cmap = plt.get_cmap('gray'))

    plt.axis('off')
    plt.title('It should be a {}.'.format(classes[1]))
plt.show()

# Scissors
plt.figure(figsize=(12,8))
for i in range(5):
    plt.subplot(1, 5, i+1)
    img = image.array_to_img(random.choice(Scissors))
    plt.imshow(img, cmap = plt.get_cmap('gray'))

    plt.axis('off')
    plt.title('It should be a {}.'.format(classes[2]))
plt.show()


## 데이터 전처리
# enumerate함수를 이용해 Rock, Paper, Scissors을 0, 1, 2로 변환
Y_Rocks, Y_Papers, Y_Scissors = [], [], []

Y_Rocks = [0 for item in enumerate(Rocks)]
Y_Papers = [1 for item in enumerate(Papers)]
Y_Scissors = [2 for item in enumerate(Scissors)]

# 리스트의 형태를 ndarray로 바꿔줌
Rocks = np.asarray(Rocks).astype('float32')
Papers = np.asarray(Papers).astype('float32')
Scissors = np.asarray(Scissors).astype('float32')

Y_Rocks = np.asarray(Y_Rocks).astype('int32')
Y_Papers = np.asarray(Y_Papers).astype('int32')
Y_Scissors = np.asarray(Y_Scissors).astype('int32')

# values값을 0과 1 사이로 맞춰줌
Rocks /= 255
Papers /= 255
Scissors /= 255

# concatenate 함수를 이용해서 배열 결합
X = np.concatenate((Rocks, Papers, Scissors), axis=0)
Y = np.concatenate((Y_Rocks, Y_Papers, Y_Scissors), axis=0)

# One-Hot Encoding
Y_encoded = np_utils.to_categorical(Y, 3)

# 학습셋, 테스트셋 구분
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded,
                                                    test_size = 0.3,
                                                    random_state = 0)

## 모델 설정
# seed값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

model = Sequential()

model.add(Conv2D(32, kernel_size = (3, 3), input_shape = (96, 96, 1), activation = 'relu'))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation = 'softmax'))

print('The model was created by following config:')
model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])

## 모델 저장 폴더 설정
Model_dir = 'Project03/model1/'
if not os.path.exists(Model_dir):
    os.mkdir(Model_dir)

## 모델 저장 조건 설정
modelpath = 'Project03/model1/{epoch:02d}-{val_loss:.4f}.hdf5'
checkpointer = ModelCheckpoint(filepath = modelpath,
                               monitor = 'val_loss',
                               verbose = 1,
                               save_best_only = True)

## 학습 자동 중단 설정
early_stopping_callback = EarlyStopping(monitor = 'val_loss',
                                        patience = 10)

## 모델 실행
history = model.fit(X, Y_encoded,
                    validation_data = (X_train, Y_train),
                    epochs = 30,
                    batch_size = 200,
                    verbose = 0,
                    callbacks = [early_stopping_callback, checkpointer])

# 테스트 정확도 출력
print('\n Test Accuracy: %.4f' % (model.evaluate(X_test, Y_test)[1]))

# 테스트셋의 오착
y_vloss = history.history['val_loss']

# 학습셋의 오착
y_loss = history.history['loss']

# loss 그래프 그리기
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker = '.', c = 'red', label = 'Testset_loss')
plt.plot(x_len, y_loss, marker = '.', c = 'blue', label = 'Trainset_loss')
plt.legend(loc = 'upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

