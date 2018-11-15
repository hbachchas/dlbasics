"""
HBImageLib
===========

Stores some image processing functionalities

For usage: browse <function_name>_usage()

"""
import numpy
from matplotlib import pyplot
from scipy.misc import toimage


def showall_components(img):
    """ Requires: import: numpy, matplotlib.pyplot, scipy.misc.toimage """
    if not (type(img) == numpy.ndarray):    # img should be a numpy array
        print('The input is not a numpy.ndarray. Terminating...')
        return
        
    img_shape = img.shape
    if len(img_shape)==2:   # for 2D image
        pyplot.imshow(toimage(img))
        pyplot.show()
    elif len(img_shape)==3:   # for 3D image
        h = img_shape[0]
        w = img_shape[1]
        data = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        for component in range(0,3):
            t_img = numpy.copy(data)
            t_img[:,:,component] = img[:,:,component]
            pyplot.subplot(331+component)
            pyplot.imshow(toimage(t_img))
        pyplot.show()

def showindividual_component(img, component):
    """ Requires: import: numpy, matplotlib.pyplot, scipy.misc.toimage """
    if not (type(img) == numpy.ndarray):    # img should be a numpy array
        print('The input is not a numpy.ndarray. Terminating...')
        return

    img_shape = img.shape
    if len(img_shape)==2:   # for 2D image
        pyplot.imshow(toimage(img))
        pyplot.show()
    elif len(img_shape)==3:   # for 3D image
        h = img_shape[0]
        w = img_shape[1]
        data = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        data[:,:,component] = img[:,:,component]
        pyplot.imshow(toimage(data))
        pyplot.show()

def showall_components_usage():
    # # Plot ad hoc CIFAR10 instances
    # from keras.datasets import cifar10
    # # load data
    # (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # showindividual_component(X_train[1],0)  # shows RED component of image X_train[1]
    # showall_components(X_train[1])      # shows all RGB componentwise results of X_train[1]

    # import matplotlib.image as mpimg
    # img=mpimg.imread('/home/himanshu/Documents/Projects/DLbasics/CNN_cifar10/data/index.jpg')   # returns a numpy.ndarray object
    # showall_components(img)     # display all individual components
    # showall_components(img[:,:,1])  # display gray scale image
    pass


