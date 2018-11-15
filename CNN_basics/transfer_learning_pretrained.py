# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/
# imagenet_mean = [103.939, 116.779, 123.68]

from keras.applications import VGG16

image_size = 224

#Load the VGG model
vgg_conv = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3))

train_dir = './train'
validation_dir = './validation'

nTrain = 600
nVal = 150

train_batch_size = 20
val_batch_size = 10

# visualizing the model
# from keras.utils import plot_model
# plot_model(vgg_conv, to_file='model.png')



######### training batch #########

# create image data generator
import keras.preprocessing.image as kpi
import numpy as np

datagen = kpi.ImageDataGenerator(rescale=1./255)
 
train_features = np.zeros(shape=(nTrain, 7, 7, 512))
train_labels = np.zeros(shape=(nTrain,3))

# read train data in batches from directory
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=train_batch_size,
    class_mode='categorical',
    shuffle=True)

# extract VGG16 features from the last layer and create training batch
i = 0
for inputs_batch, labels_batch in train_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    train_features[i * train_batch_size : (i + 1) * train_batch_size] = features_batch
    train_labels[i * train_batch_size : (i + 1) * train_batch_size] = labels_batch
    i += 1
    if i * train_batch_size >= nTrain:
        break

# linearize all the features
train_features = np.reshape(train_features, (nTrain, 7 * 7 * 512))



######### validation batch #########

# create image data generator
validation_features = np.zeros(shape=(nVal, 7, 7, 512))
validation_labels = np.zeros(shape=(nVal,3))

# read validation data in batches from directory
validation_generator = datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=val_batch_size,
    class_mode='categorical',
    shuffle=True)

# extract VGG16 features from the last layer and create validation batch
i = 0
for inputs_batch, labels_batch in validation_generator:
    features_batch = vgg_conv.predict(inputs_batch)
    validation_features[i * val_batch_size : (i + 1) * val_batch_size] = features_batch
    validation_labels[i * val_batch_size : (i + 1) * val_batch_size] = labels_batch
    i += 1
    if i * val_batch_size >= nVal:
        break

# linearize all the features
validation_features = np.reshape(validation_features, (nVal, 7 * 7 * 512))



# create a neural network and train with the extracted features
from keras import models
from keras import layers
from keras import optimizers
# define model
model = models.Sequential()
model.add(layers.Dense(256, activation='relu', input_dim=7 * 7 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))
# compile model
model.compile(
    optimizer=optimizers.RMSprop(lr=2e-4),
    loss='categorical_crossentropy',
    metrics=['acc'])
# train model
history = model.fit(
    train_features,
    train_labels,
    epochs=20,
    batch_size=val_batch_size,      # val_batch_size is factor of train_batch_size
    validation_data=(validation_features,validation_labels))