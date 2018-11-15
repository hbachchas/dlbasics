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

def load_dataset():
    """ load [X, y] """
    dirHC = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/HC'     # positive
    dirIC = '/home/peace/Documents/z440/DLBasics/Anjali/data/rgbData/IC'     # negative
    filesPos = os.listdir(dirHC)
    filesNeg = os.listdir(dirIC)

    numpy.random.shuffle(filesPos)
    numpy.random.shuffle(filesPos)
    numpy.random.shuffle(filesNeg)
    numpy.random.shuffle(filesNeg)

    totalfiles = len(filesPos) + len(filesNeg)
    X = numpy.zeros( (totalfiles,224,224,3), dtype=numpy.uint8 )
    y = numpy.zeros( totalfiles, dtype=numpy.uint8 )

    # load train data
    currentIdx = 0
    print('Loading train data...')
    while( len(filesPos) + len(filesNeg) ):
        show_progress( max_val=totalfiles, present_val=totalfiles-len(filesPos)-len(filesNeg) )
        if len(filesPos):
            impath = filesPos.pop()
            im1 = cv2.imread( path.join(dirHC,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y[currentIdx] = 1
            currentIdx = currentIdx + 1
        if len(filesNeg):
            impath = filesNeg.pop()
            im1 = cv2.imread( path.join(dirIC,impath) )
            assert im1.shape[2] == 3    # 3 channel assertion
            X[currentIdx,:,:,:] = cv2.resize( im1, dsize=(224,224), interpolation=cv2.INTER_CUBIC )
            y[currentIdx] = 0
            currentIdx = currentIdx + 1

    return [X, y]

# if __name__ == '__main__':
#     X, y = load_dataset()
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


