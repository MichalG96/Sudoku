# Script for training neural network in recognizing digits,
# using MNIST and Char74k datasets

from __future__ import print_function
import os
import cv2
import numpy as np
import PIL.ImageOps
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import sys

BATCH_SIZE = 128
NUM_CLASSES = 9
EPOCHS = 12

# input image dimensions
img_rows, img_cols = 28, 28

current_dir = os.getcwd()
default_dir = current_dir
# Path to folder containing training samples.
# Folder contains 9 subfolders, named 1-9, each contains 4 816 samples of a digit
relative_path = 'datasets/dataset_74k_UCI'
dataset_dir = os.path.join(current_dir, relative_path)
np.set_printoptions(threshold=sys.maxsize)

x = []
y = []

for folder_name in os.listdir(relative_path):
    digit = int(folder_name)
    os.chdir(os.path.join(dataset_dir, folder_name))
    dir_list = os.listdir(os.getcwd())
    y += len(dir_list) * [digit]
    for file in dir_list:
        im = cv2.imread(f'{file}', cv2.IMREAD_GRAYSCALE)
        x.append(cv2.bitwise_not(im))

x = np.asarray(x)
print(f'x shape: {x.shape}')
y = np.asarray(y)

def unison_shuffling(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

x, y = unison_shuffling(x, y)

# Split dataset into training and testing samples
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

ind_train = np.argsort(y_train_mnist)
ind_test = np.argsort(y_test_mnist)

x_train_mnist = x_train_mnist[ind_train][6000:]
y_train_mnist = y_train_mnist[ind_train][6000:]
x_test_mnist = x_test_mnist[ind_test][1000:]
y_test_mnist = y_test_mnist[ind_test][1000:]

training_samples = 36000

x_train = x[:training_samples]
x_test = x[training_samples:]
y_train = y[:training_samples]
y_test = y[training_samples:]

x_train = np.concatenate((x_train, x_train_mnist))
y_train = np.concatenate((y_train, y_train_mnist))-1
x_test = np.concatenate((x_test, x_test_mnist))
y_test = np.concatenate((y_test, y_test_mnist))-1

x_train, y_train = unison_shuffling(x_train, y_train)
x_test, y_test = unison_shuffling(x_test, y_test)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print(f'hmm: {(sorted(y_train))}')
print(f'hmm: {(y_test.shape)}')

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()
model.add(Conv2D(20, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

os.chdir(default_dir)
filename = r'weights_{loss:.4f}-{epoch:02d}.hdf5'
save_weights_path = os.path.join('weights', filename)
checkpoint = ModelCheckpoint(save_weights_path, monitor='loss', verbose=1, save_best_only=True, mode='min')
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_path = 'weights/MNIST_74k_UCI.h5'
model.save(model_path)

# Check the model on some random data
relative_path = 'datasets/evaluation_set'

for filename in os.listdir(relative_path):
    test_model = load_model(model_path)
    img = load_img(f'{relative_path}/{filename}', color_mode='grayscale',
                   target_size=(28, 28))

    input_eval = PIL.ImageOps.invert(img)
    input_eval = img_to_array(input_eval)
    input_eval = input_eval.astype('float32')
    input_eval /= 255
    input_eval = np.expand_dims(input_eval, axis=0)
    predicted_class = test_model.predict_classes(input_eval)
    prob = test_model.predict(input_eval)
    print(filename, predicted_class, prob)


