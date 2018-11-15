# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
# imagenet_mean = [103.939, 116.779, 123.68]
# Four things were observed :
# model -- matters most
# batchsize -- convergence is delayed for smaller batch size, larger batch size is better
# learning rate -- .0001 lesser fluctuations (smoother) + faster training than .001
# dropout -- delayed convergence, around .2 is better than .5, needs more number of epochs
# 
# preprocessing -- scale and center the data
# layers should not be freezed

from keras.applications import vgg16
from keras.applications.imagenet_utils import decode_predictions
from keras import layers
from keras import models
from keras import optimizers
import keras.backend as K
from keras.callbacks import CSVLogger
import numpy
from os import path
import os, seaborn, csv, math, shutil
from time import time
from load_data import load_dataset

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)

def savePlots(history, savedir):
    # visualize the training progress     # keep verbose=1 in model.fit()
    #%matplotlib inline
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure, draw
    # Plot training & validation accuracy values
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()
    plt.savefig(path.join(savedir,'Acc.eps'), format='eps', dpi=400)
    # Plot training & validation loss values
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='lower left')
    # plt.show()
    plt.savefig(path.join(savedir,'Loss.eps'), format='eps', dpi=400)
    print('Plots saved to disk')

def evaluateModel(X_test, y_test, model):
    """
    Final evaluation of the model
    """
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    y_predicted = []
    y_probability = []

    for i in range(len(y_test)):
        inp = X_test[i,:,:,:]
        inp_batch = numpy.expand_dims(inp,axis=0)

        pred = model.predict(inp_batch,verbose=1)
        probability = pred[0][0]
        y_probability.append(probability)
        
        classLabel = model.predict_classes(inp_batch,verbose=1)
        predicted_label = classLabel[0][0]
        y_predicted.append(predicted_label)

    return [y_predicted, y_probability, scores[1]*100]

def savePredictions2disk(y_test, y_predicted, y_probability, savedir):   
    assert len(y_test) == len(y_predicted)
    myData = []
    temp = ['Actual','Predicted','Probability/Score']
    myData.append(temp)
    for i in range( len(y_test) ):
        temp = []
        temp.append(y_test[i])
        temp.append(y_predicted[i])
        temp.append(y_probability[i])
        myData.append(temp)

    myFile = open(path.join(savedir,'Predictions.csv'), 'w')  
    with myFile:  
        writer = csv.writer(myFile)
        writer.writerows(myData)

def saveModelArchWeights(model, savedir):
    """
    save model architecture and weights
    """
    # serialize model to JSON
    model_json = model.to_json()
    with open(path.join(savedir,'model.json'), "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(path.join(savedir,'model.h5'))
    print("Model saved to disk")

def defineModel():
    """
    1. load ResNet50 model without top
    3. add dense layers
    """
    # load the VGG model
    vgg_conv = vgg16.VGG16( weights='imagenet', include_top=False, input_shape=(224, 224, 3) )
    
    # # freeze the layers except the last 4 layers
    # for layer in resnet50_conv.layers[:-4]:
    #    layer.trainable = False

    # # check the trainable status of the individual layers
    # for layer in resnet50_conv.layers:
    #     print(layer, layer.trainable)

    # create the model
    model = models.Sequential()

    # add the ResNet50 convolutional base model
    model.add(vgg_conv)

    # add new layers
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    # show a summary of the model. Check the number of trainable parameters
    model.summary()

    return model

def preprocessDataset(X_train, X_test):
    """
    prepare the image for the ResNet50 model
    """
    X_train = X_train.astype('float32')
    X_train = X_train / 255.0
    X_train[:,:,:,:] -= 0.5
    
    X_test = X_test.astype('float32')
    X_test = X_test / 255.0
    X_test[:,:,:,:] -= 0.5

    return [X_train, X_test]

def createLogFile(evaluation_dir):
    if os.path.isfile( path.join(evaluation_dir,'log.csv') ):
        os.remove( path.join(evaluation_dir,'log.csv') )
    f = open( path.join(evaluation_dir,'log.csv'), "x" )
    f.close()

def loadModelArchWeights(loaddir):
    """
    load the model architecture and weights
    """
    from keras.models import model_from_json
    # load json and create model
    json_file = open( path.join(loaddir,'model.json'), 'r' )
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights( path.join(loaddir,'model.h5') )
    print("Model loaded from disk")
    return loaded_model

def divideDataIntoKfolds(X, y, k_fold):
    """
    returns the list of (X/y train/test) indices for every fold
    
    this function can handle non perfect len(X)
    """
    assert len(X) == len(y)
    X_train_arr = []
    y_train_arr = []
    X_test_arr = []
    y_test_arr = []
    total_elements = len(X)
    elements_per_fold = math.floor( total_elements / k_fold )
    x_temp = [i for i in range(len(X))]

    for k in range(k_fold):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        if k+1 == k_fold:    # for the last fold
            X_test += x_temp[ 0:k*elements_per_fold ]
            y_test += x_temp[ 0:k*elements_per_fold ]
            X_train += x_temp[ k*elements_per_fold:len(x_temp) ]
            y_train += x_temp[ k*elements_per_fold:len(x_temp) ]
        else:
            X_test += x_temp[ 0:k*elements_per_fold ]
            y_test += x_temp[ 0:k*elements_per_fold ]
            X_train += x_temp[ k*elements_per_fold:(k+1)*elements_per_fold ]
            y_train += x_temp[ k*elements_per_fold:(k+1)*elements_per_fold ]
            X_test += x_temp[ (k+1)*elements_per_fold:len(x_temp) ]
            y_test += x_temp[ (k+1)*elements_per_fold:len(x_temp) ]
        X_train_arr.append(X_train)
        y_train_arr.append(y_train)
        X_test_arr.append(X_test)
        y_test_arr.append(y_test)

    return [X_train_arr, y_train_arr, X_test_arr, y_test_arr]

def getDataPerFold(X_train_arr, y_train_arr, X_test_arr, y_test_arr, X, y, k):
    X_train = numpy.zeros( (len(X_train_arr[k]),224,224,3), dtype=numpy.uint8 )
    y_train = numpy.zeros( len(y_train_arr[k]), dtype=numpy.uint8 )
    X_test = numpy.zeros( (len(X_test_arr[k]),224,224,3), dtype=numpy.uint8 )
    y_test = numpy.zeros( len(y_test_arr[k]), dtype=numpy.uint8 )

    idx_counter = 0
    for idx in X_train_arr[k]:
        X_train[idx_counter,:,:,:] = X[idx,:,:,:]
        y_train[idx_counter] = y[idx]
        idx_counter += 1
    idx_counter = 0
    for idx in X_test_arr[k]:
        X_test[idx_counter,:,:,:] = X[idx,:,:,:]
        y_test[idx_counter] = y[idx]
        idx_counter += 1

    # changing the order of train and test
    return [X_test, y_test, X_train, y_train]

def runFromScratch():
    # fix random seed for reproducibility
    seed = 7
    numpy.random.seed(seed)

    evaluation_dir = '/home/peace/Documents/z440/DLBasics/Anjali/experiments/e10fold/evaluation'
    
    print('Loading dataset...')
    X, y = load_dataset()
    # seaborn.countplot(y_train)
    # seaborn.countplot(y_test)
    
    k_fold = 10
    score_arr = []
    X_train_arr, y_train_arr, X_test_arr, y_test_arr = divideDataIntoKfolds(X, y, k_fold)       # get indices dividing data into 10 sets

    for k in range(k_fold):     # repeat for each fold
        print('Executing fold number : ', k+1)
        X_train, y_train, X_test, y_test = getDataPerFold( X_train_arr, y_train_arr, X_test_arr, y_test_arr, X, y, k )
        current_evaluation_dir = path.join(evaluation_dir,str(k+1))
        if os.path.isdir( current_evaluation_dir ):
            shutil.rmtree( current_evaluation_dir )
        os.mkdir( current_evaluation_dir )

        # prepare the image for the ResNet50 model
        X_train, X_test = preprocessDataset(X_train, X_test)

        model = defineModel()

        # compile model
        optimizr = optimizers.Adam(lr=0.001)
        model.compile( loss='binary_crossentropy', optimizer=optimizr, metrics=['accuracy'] )
        # model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', mean_pred] )

        # fit model
        numEpochs = 30
        batchSize = 12
        print('Fitting the model... with lr = 0.001')
        createLogFile(current_evaluation_dir)
        csv_logger = CSVLogger(path.join(current_evaluation_dir,'log.csv'), append=True, separator=';')
        start_time = time()
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=numEpochs, batch_size=batchSize, verbose=1, callbacks=[csv_logger])
        print( 'Training time = ',time()-start_time )
        # save model architecture and weights
        saveModelArchWeights(model, current_evaluation_dir)

        # save plots to disk
        savePlots(history, current_evaluation_dir)

        # evaluate model & print accuracy
        [y_predicted, y_probability, scores] = evaluateModel(X_test, y_test, model)

        # save predictions to disk
        savePredictions2disk(y_test, y_predicted, y_probability, current_evaluation_dir)

        score_arr.append(scores)
        # # get the predicted probabilities for each class
        # predictions = model.predict(X_test)
        # print('predictions size', predictions.shape)
        
        # # convert the probabilities to class labels
        # # We will get top 5 predictions which is the default
        # label = decode_predictions(predictions, top=2)
        # print(label)
    print ( 'Average accuracy = ', sum(score_arr)/len(score_arr) )

# def runSavedModel():
#     # fix random seed for reproducibility
#     seed = 7
#     numpy.random.seed(seed)

#     evaluation_dir = '/home/himanshu/Anjali/experiments/e10fold/evaluation'

#     print('Loading dataset...')
#     X, y = load_dataset()
#     # seaborn.countplot(y_train)
#     # seaborn.countplot(y_test)
    
#     # prepare the image for the ResNet50 model
#     X_train, X_test = preprocessDataset(X_train, X_test)

#     # load the model architecture and weights
#     model = loadModelArchWeights(evaluation_dir)

#     # compile model
#     optimizr = optimizers.Adam(lr=0.001)
#     model.compile( loss='binary_crossentropy', optimizer=optimizr, metrics=['accuracy'] )
#     # model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', mean_pred] )

#     # # fit model
#     # numEpochs = 15
#     # batchSize = 6
#     # print('Fitting the model...')
#     # createLogFile(evaluation_dir)
#     # csv_logger = CSVLogger(path.join(evaluation_dir,'log.csv'), append=True, separator=';')
#     # history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=numEpochs, batch_size=batchSize, verbose=1, callbacks=[csv_logger])
#     # # save model architecture and weights
#     # saveModelArchWeights(model, evaluation_dir)
#     # # save plots to disk
#     # savePlots(history, evaluation_dir)

#     # evaluate model & print accuracy
#     [y_predicted, y_probability, scores] = evaluateModel(X_test, y_test, model)

#     # save predictions to disk
#     savePredictions2disk(y_test, y_predicted, y_probability, evaluation_dir)

#     # # get the predicted probabilities for each class
#     # predictions = model.predict(X_test)
#     # print('predictions size', predictions.shape)
    
#     # # convert the probabilities to class labels
#     # # We will get top 5 predictions which is the default
#     # label = decode_predictions(predictions, top=2)
#     # print(label)

if __name__ == '__main__':
    
    # runSavedModel()
    runFromScratch()
 