'''
File to handle data manipulation utility functions.
Ex train/test splits and other stuff.
'''

import os, sys
import numpy as np
import pickle
from scipy.ndimage import imread

DEBUG = 3 
DATA_FILE = './data/Training/'

def load_train_data():

    class_data = {} 
    skip_first = True 
    for temp in os.walk(DATA_FILE):

        # we don't want this first entry
        if skip_first:
            skip_first = False
            continue

        # add class name to dictionary
        class_name = temp[0].split('/')[-1]
        class_data[class_name] = [] 

        if DEBUG > 2:
            sys.stdout.write('\rloading training data class {} of 60...'.format(len(class_data)))
            sys.stdout.flush()
            
        # add image data to dictionary
        for file_name in temp[2]:
            real_dir = os.path.join(DATA_FILE, class_name, file_name)
            img_data = imread(real_dir)
            class_data[class_name].append(img_data)

        class_data[class_name] = np.array(class_data[class_name])

    if DEBUG > 2:
        print ' done!\nloaded:'
        print '\t{} classes'.format(len(class_data))
        print '\t{} images per class\n'.format(len(class_data.itervalues().next()))

    return class_data
