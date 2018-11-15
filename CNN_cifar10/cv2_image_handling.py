# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.utils import np_utils

import cv2

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# img = cv2.imread('./data/index.jpg')
img = X_train[1]
res = cv2.resize(img, dsize=(420, 240), interpolation=cv2.INTER_CUBIC)      # Resize image

# Display images.
cv2.imshow("Original Image", img)
cv2.imshow("Resized Image", res)

print(type(img))
print(type(res))
print(img.shape)
print(res.shape)
cv2.waitKey(0)

t_X_train = X_train.copy()
t_X_test = X_test.copy()

X_train = numpy.zeros((10,24,42,3), dtype=numpy.uint8)

for i in range(0,10):
    img = t_X_train[i]
    X_train[i,:,:,:] = cv2.resize(img, dsize=(42, 24), interpolation=cv2.INTER_CUBIC)      # Resize image

for i in range(0,10):
    cv2.imshow("Original Image", X_train[i])
    cv2.waitKey(0)