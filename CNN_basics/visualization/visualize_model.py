# https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/
# imagenet_mean = [103.939, 116.779, 123.68]

from keras.applications import VGG16

#Load the VGG model
vgg_conv = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3))

train_dir = './train'
validation_dir = './validation'

nTrain = 600
nVal = 150

# visualizing the model
from keras.utils import plot_model
plot_model(vgg_conv, to_file='model.png')

# Show a summary of the model. Check the number of trainable parameters
vgg_conv.summary()