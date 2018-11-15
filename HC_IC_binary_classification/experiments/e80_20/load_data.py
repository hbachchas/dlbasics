import os
from os import path
import numpy
import cv2
import math
import sys
import scipy.io

# add local path
sys.path.append('/home/peace/Documents/z440/DLBasics/GIT/exp')
import pythonhblib.randhb as rhb

def show_progress(max_val, present_val):
    progress = present_val / (max_val+1) * 100
    sys.stdout.write("Progress: %d%%   \r" % (progress) )
    sys.stdout.flush()

def load_partitioned_dataset():
    dirTrainPos = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/train/HC'
    dirTrainNeg = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/train/IC'
    dirTestPos = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/test/HC'
    dirTestNeg = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/test/IC'

    trainPos = os.listdir(dirTrainPos)
    trainNeg = os.listdir(dirTrainNeg)
    testPos = os.listdir(dirTestPos)
    testNeg = os.listdir(dirTestNeg)

    cnt_train = len(trainPos) + len(trainNeg)
    X_train = numpy.zeros( (cnt_train,224,224,3), dtype=numpy.uint8 )
    y_train = numpy.zeros( cnt_train, dtype=numpy.uint8 )
    cnt_test = len(testPos) + len(testNeg)
    X_test = numpy.zeros( (cnt_test,224,224,3), dtype=numpy.uint8 )
    y_test = numpy.zeros( cnt_test, dtype=numpy.uint8 )

    # load train data
    currentIdx = 0
    print('Loading train data...')
    while( len(trainPos) + len(trainNeg) ):
        show_progress( max_val=cnt_train, present_val=cnt_train-len(trainPos)-len(trainNeg) )
        if len(trainPos):
            impath = trainPos.pop()
            im1 = cv2.imread( path.join(dirTrainPos,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_train[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_train[currentIdx] = 1
            currentIdx = currentIdx + 1
        if len(trainNeg):
            impath = trainNeg.pop()
            im1 = cv2.imread( path.join(dirTrainNeg,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_train[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_train[currentIdx] = 0
            currentIdx = currentIdx + 1

    # load test data
    currentIdx = 0
    print('Loading test data...')
    while( len(testPos) + len(testNeg) ):
        show_progress( max_val=cnt_test, present_val=cnt_test-len(testPos)-len(testNeg) )
        if len(testPos):
            impath = testPos.pop()
            im1 = cv2.imread( path.join(dirTestPos,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_test[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_test[currentIdx] = 1
            currentIdx = currentIdx + 1
        if len(testNeg):
            impath = testNeg.pop()
            im1 = cv2.imread( path.join(dirTestNeg,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_test[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_test[currentIdx] = 0
            currentIdx = currentIdx + 1

    return [X_train, y_train, X_test, y_test]

def load_dataset(threshold=0.2):
    dirHC = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/HC'     # positive
    dirIC = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/IC'     # negative
    filesPos = os.listdir(dirHC)
    filesNeg = os.listdir(dirIC)

    trainPos, testPos = rhb.get_rand_list(filesPos,th=threshold)
    trainNeg, testNeg = rhb.get_rand_list(filesNeg,th=threshold)

    cnt_train = len(trainPos) + len(trainNeg)
    X_train = numpy.zeros( (cnt_train,224,224,3), dtype=numpy.uint8 )
    y_train = numpy.zeros( cnt_train, dtype=numpy.uint8 )
    cnt_test = len(testPos) + len(testNeg)
    X_test = numpy.zeros( (cnt_test,224,224,3), dtype=numpy.uint8 )
    y_test = numpy.zeros( cnt_test, dtype=numpy.uint8 )

    # load train data
    currentIdx = 0
    print('Loading train data...')
    while( len(trainPos) + len(trainNeg) ):
        show_progress( max_val=cnt_train, present_val=cnt_train-len(trainPos)-len(trainNeg) )
        if len(trainPos):
            impath = trainPos.pop()
            im1 = cv2.imread( path.join(dirHC,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_train[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_train[currentIdx] = 1
            currentIdx = currentIdx + 1
        if len(trainNeg):
            impath = trainNeg.pop()
            im1 = cv2.imread( path.join(dirIC,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_train[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_train[currentIdx] = 0
            currentIdx = currentIdx + 1

    # load test data
    currentIdx = 0
    print('Loading test data...')
    while( len(testPos) + len(testNeg) ):
        show_progress( max_val=cnt_test, present_val=cnt_test-len(testPos)-len(testNeg) )
        if len(testPos):
            impath = testPos.pop()
            im1 = cv2.imread( path.join(dirHC,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_test[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_test[currentIdx] = 1
            currentIdx = currentIdx + 1
        if len(testNeg):
            impath = testNeg.pop()
            im1 = cv2.imread( path.join(dirIC,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X_test[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y_test[currentIdx] = 0
            currentIdx = currentIdx + 1

    return [X_train, y_train, X_test, y_test]





# if __name__ == '__main__':
#     X_train, y_train, X_test, y_test = load_partitioned_dataset()
#     print('asdf')

#     print ('Dataset loaded.')
#     bool_var = True
#     # while bool_var:
#     #     i = input()
#     i = 10
#     cv2.imshow("Image 1", X_train[i,:,:,:])
#     cv2.imshow("Image 2", X_train[i+1,:,:,:])
#     cv2.imshow("Image 3", X_train[i+2,:,:,:])
#     cv2.imshow("Image 4", X_train[i+3,:,:,:])
#     cv2.imshow("Image 5", X_train[i+4,:,:,:])
#     print(y_train[i], y_train[i+1], y_train[i+2], y_train[i+3], y_train[i+4])
#     # cv2.imshow("Image 1", X_train[int(i),:,:,:])
#     # cv2.imshow("Image 2", X_train[int(i+1),:,:,:])
#     # cv2.imshow("Image 3", X_train[int(i+2),:,:,:])
#     # cv2.imshow("Image 4", X_train[int(i+3),:,:,:])
#     # cv2.imshow("Image 5", X_train[int(i+4),:,:,:])
#     # print(y_train[int(i)], y_train[int(i+1)], y_train[int(i+2)], y_train[int(i+3)], y_train[int(i+4)])
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()    # without this jupyter notebook will crash


