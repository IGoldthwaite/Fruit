'''
File to handle data manipulation utility functions.
Ex train/test splits and other stuff.
'''

import os, sys
import numpy as np
import pickle
import cv2

DEBUG = 3 
TRAIN_DATA_DIRECTORY = './data/Training/'
TEST_DATA_DIRECTORY = './data/Validation/'

def load_data(train, class_limit=-1):

    print 'loading training data' if train else 'loading test data'
    directory = TRAIN_DATA_DIRECTORY if train else TEST_DATA_DIRECTORY
    X = []
    y = [] 
    skip_first = True 

    class_count = 0

    for temp in os.walk(directory):

        if class_limit > 0 and class_count > class_limit:
            break
        else:
            class_count += 1

        # we don't want this first entry
        if skip_first:
            skip_first = False
            continue

        # add class name to dictionary
        class_name = temp[0].split('/')[-1]

        if DEBUG > 2:
            sys.stdout.write('\rloading data class for class {}                        \r'.format(class_name))
            sys.stdout.flush()
            
        # add image data to dictionary
        for file_name in temp[2]:
            real_dir = os.path.join(directory, class_name, file_name)
            image = cv2.imread(real_dir, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (45, 45))
            
            # commenting this out allows us to reconstruct images by simply reshaping
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            image = image.flatten()
            X.append(image)
            y.append(class_name)

    class_id_map = {v:i for i,v in enumerate(np.unique(y))}

    X = np.array(X)
    y = np.array(y)

    if DEBUG > 2:
        print ''
        print ' done!\nloaded:'
        print '\t{} classes'.format(len(class_id_map))
        print '\t{} data entries\n'.format(len(X))

    return X, y

def print_accuracy(predictions, labels):
    num_correct = 0
    num_error = 0
    for pi in range(len(predictions)):
        if predictions[pi] != labels[pi]:
            num_error += 1
        else:
            num_correct += 1
    print '{} of {} correct'.format(num_correct, num_correct+num_error)
    print 'Accuracy: {}%'.format(float(num_correct*100)/float(num_correct+num_error))
