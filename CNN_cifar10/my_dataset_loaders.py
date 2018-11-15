import os
from os import path
import numpy
import cv2
import math
import sys
import scipy.io

def show_progress(max_val, present_val):
    progress = present_val / (max_val+1) * 100
    sys.stdout.write("Progress: %d%%   \r" % (progress) )
    sys.stdout.flush()

def load_dataset_RGB(split_th = 0.8, ext='.jpg'):
    """ Default: 80% for training, 20% for testing """

    positive_dir = '/media/himanshu/ce640fc3-0289-402c-9150-793e07e55b8c/visapp2018code/RGB/data/positive'
    negative_dir = '/media/himanshu/ce640fc3-0289-402c-9150-793e07e55b8c/visapp2018code/RGB/data/negative'
    # positive_dir = '/home/himanshu/Documents/Projects/DLbasics/visapp2018code/RGB/data/positive'
    # negative_dir = '/home/himanshu/Documents/Projects/DLbasics/visapp2018code/RGB/data/negative'
    t_files = os.listdir(path.join(positive_dir, '1'))
    total_pos_files = len(t_files)
    t_files = os.listdir(path.join(negative_dir, '1'))
    total_neg_files = len(t_files)
    print('pos files: ',total_pos_files)
    print('neg files: ',total_neg_files)
    
    # total_files = total_pos_files + total_neg_files
    total_files = 1000

    X1 = numpy.zeros( (total_files,96,128,3), dtype=numpy.uint8 )
    X2 = numpy.zeros( (total_files,96,128,3), dtype=numpy.uint8 )
    X3 = numpy.zeros( (total_files,96,128,3), dtype=numpy.uint8 )
    y = numpy.zeros( (total_files), dtype=numpy.uint8 )

    pos_file_counter = 0
    neg_file_counter = 0
    total_counter = 0
    while total_counter < total_files:
        show_progress(max_val=total_files, present_val=total_counter)
        if total_counter % 2 == 0:   # case: positive
            im1_path = path.join(positive_dir, '1', str(pos_file_counter+1)+ext)
            im2_path = path.join(positive_dir, '2', str(pos_file_counter+1)+ext)
            im3_path = path.join(positive_dir, '3', str(pos_file_counter+1)+ext)
            im1 = cv2.imread(im1_path)
            im2 = cv2.imread(im2_path)
            im3 = cv2.imread(im3_path)

            # cv2.imshow("Image 1", im1)
            # cv2.imshow("Image 2", im2)
            # cv2.imshow("Image 3", im3)
            # cv2.waitKey(0)

            X1[total_counter,:,:,:] = cv2.resize(im1, dsize=(128, 96), interpolation=cv2.INTER_CUBIC)      # Resize image
            X2[total_counter,:,:,:] = cv2.resize(im2, dsize=(128, 96), interpolation=cv2.INTER_CUBIC)      # Resize image
            X3[total_counter,:,:,:] = cv2.resize(im3, dsize=(128, 96), interpolation=cv2.INTER_CUBIC)      # Resize image
            y[total_counter] = 1
            pos_file_counter += 1
        else:
            im1_path = path.join(negative_dir, '1', str(neg_file_counter+1)+ext)
            im2_path = path.join(negative_dir, '2', str(neg_file_counter+1)+ext)
            im3_path = path.join(negative_dir, '3', str(neg_file_counter+1)+ext)
            im1 = cv2.imread(im1_path)
            im2 = cv2.imread(im2_path)
            im3 = cv2.imread(im3_path)

            # cv2.imshow("Image 1", im1)
            # cv2.imshow("Image 2", im2)
            # cv2.imshow("Image 3", im3)
            # cv2.waitKey(0)

            X1[total_counter,:,:,:] = cv2.resize(im1, dsize=(128, 96), interpolation=cv2.INTER_CUBIC)      # Resize image
            X2[total_counter,:,:,:] = cv2.resize(im2, dsize=(128, 96), interpolation=cv2.INTER_CUBIC)      # Resize image
            X3[total_counter,:,:,:] = cv2.resize(im3, dsize=(128, 96), interpolation=cv2.INTER_CUBIC)      # Resize image            
            y[total_counter] = 0
            neg_file_counter += 1

        total_counter += 1
    
    # normalize inputs from 0-255 to 0.0-1.0
    X1 = X1.astype('float32')
    X2 = X2.astype('float32')
    X3 = X3.astype('float32')
    X1 = X1 / 255.0
    X2 = X2 / 255.0
    X3 = X3 / 255.0

    training_samples_limit = math.ceil( split_th * total_counter )
    X1_train = X1[0:training_samples_limit,:,:,:]
    X2_train = X2[0:training_samples_limit,:,:,:]
    X3_train = X3[0:training_samples_limit,:,:,:]
    y_train = y[0:training_samples_limit]

    X1_test = X1[training_samples_limit:total_counter,:,:,:]
    X2_test = X2[training_samples_limit:total_counter,:,:,:]
    X3_test = X3[training_samples_limit:total_counter,:,:,:]
    y_test = y[training_samples_limit:total_counter]

    return [X1_train, X2_train, X3_train, y_train, X1_test, X2_test, X3_test, y_test]

def load_dataset_OF_BW(split_th = 0.8, ext='.mat'):
    """ Default: 80% for training, 20% for testing """

    positive_dir = '/home/himanshu/Documents/Projects/DLbasics/slink/BW_OF/data/positive'
    negative_dir = '/home/himanshu/Documents/Projects/DLbasics/slink/BW_OF/data/negative'
    # positive_dir = '/home/himanshu/Documents/Projects/DLbasics/visapp2018code/RGB/data/positive'
    # negative_dir = '/home/himanshu/Documents/Projects/DLbasics/visapp2018code/RGB/data/negative'
    t_files = os.listdir(path.join(positive_dir, '1'))
    total_pos_files = len(t_files)
    t_files = os.listdir(path.join(negative_dir, '1'))
    total_neg_files = len(t_files)
    print('pos files: ',total_pos_files)
    print('neg files: ',total_neg_files)
    
    # total_files = total_pos_files + total_neg_files
    total_files = 1500

    X1 = numpy.zeros( (total_files,96,128,3), dtype=numpy.float32 )
    X2 = numpy.zeros( (total_files,96,128,3), dtype=numpy.float32 )
    X3 = numpy.zeros( (total_files,96,128,3), dtype=numpy.float32 )
    y = numpy.zeros( (total_files), dtype=numpy.uint8 )

    pos_file_counter = 0
    neg_file_counter = 0
    total_counter = 0
    while total_counter < total_files:
        show_progress(max_val=total_files, present_val=total_counter)
        if total_counter % 2 == 0:   # case: positive
            im1_path = path.join(positive_dir, '1', str(pos_file_counter+1)+ext)
            im2_path = path.join(positive_dir, '2', str(pos_file_counter+1)+ext)
            im3_path = path.join(positive_dir, '3', str(pos_file_counter+1)+ext)
            mat1 = scipy.io.loadmat(im1_path)['flow']
            mat2 = scipy.io.loadmat(im2_path)['flow']
            mat3 = scipy.io.loadmat(im3_path)['flow']

            # cv2.imshow("Image 1", im1)
            # cv2.imshow("Image 2", im2)
            # cv2.imshow("Image 3", im3)
            # cv2.waitKey(0)

            X1[total_counter,:,:,:] = mat1
            X2[total_counter,:,:,:] = mat2
            X3[total_counter,:,:,:] = mat3
            y[total_counter] = 1
            pos_file_counter += 1
        else:
            im1_path = path.join(negative_dir, '1', str(neg_file_counter+1)+ext)
            im2_path = path.join(negative_dir, '2', str(neg_file_counter+1)+ext)
            im3_path = path.join(negative_dir, '3', str(neg_file_counter+1)+ext)
            mat1 = scipy.io.loadmat(im1_path)['flow']
            mat2 = scipy.io.loadmat(im2_path)['flow']
            mat3 = scipy.io.loadmat(im3_path)['flow']

            # cv2.imshow("Image 1", im1)
            # cv2.imshow("Image 2", im2)
            # cv2.imshow("Image 3", im3)
            # cv2.waitKey(0)

            X1[total_counter,:,:,:] = mat1
            X2[total_counter,:,:,:] = mat2
            X3[total_counter,:,:,:] = mat3
            y[total_counter] = 0
            neg_file_counter += 1

        total_counter += 1
    
    training_samples_limit = math.ceil( split_th * total_counter )
    X1_train = X1[0:training_samples_limit,:,:,:]
    X2_train = X2[0:training_samples_limit,:,:,:]
    X3_train = X3[0:training_samples_limit,:,:,:]
    y_train = y[0:training_samples_limit]

    X1_test = X1[training_samples_limit:total_counter,:,:,:]
    X2_test = X2[training_samples_limit:total_counter,:,:,:]
    X3_test = X3[training_samples_limit:total_counter,:,:,:]
    y_test = y[training_samples_limit:total_counter]

    return [X1_train, X2_train, X3_train, y_train, X1_test, X2_test, X3_test, y_test]

if __name__ == '__main__':
    X1_train, X2_train, X3_train, y_train, X1_test, X2_test, X3_test, y_test = load_dataset_OF_BW()

#     print ('Dataset loaded.')
#     bool_var = True
#     while bool_var:
#         i = input()
        
#         cv2.imshow("Image X1", X1_train[int(i),:,:,0])
#         cv2.imshow("Image X2", X2_train[int(i),:,:,0])
#         cv2.imshow("Image X3", X3_train[int(i),:,:,0])
#         print(y_train[int(i)])
#         cv2.waitKey(0)


