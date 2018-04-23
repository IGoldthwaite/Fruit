'''
File to handle data manipulation utility functions.
Ex train/test splits and other stuff.
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

import os, sys
import numpy as np
import pickle
import cv2

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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

def apply_pca(train_data, test_data, dimensions):
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)
    test_data_scaled = scaler.transform(test_data)
    pcadecomp = PCA(n_components=dimensions)
    pca_train = pcadecomp.fit_transform(train_data_scaled)
    pca_test = pcadecomp.transform(test_data_scaled)
    return pca_train, pca_test

def plot_pca(data, labels):
    print 'Plotting PCA in 3 dimensions'
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pcadecomp = PCA(n_components=3)
    pca_data = pcadecomp.fit_transform(data_scaled)
    pca_data = np.array(pca_data)

    classes = list(set(labels))
    pca_by_class = {} 
    for cl in classes:
        pca_by_class[cl] = np.array([pca_data[i] for i in range(len(pca_data)) if labels[i] == cl])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for cl in pca_by_class:
        ax.scatter(pca_by_class[cl].T[0], pca_by_class[cl].T[1], pca_by_class[cl].T[2]) 

    print 'Saving PCA results to figures/pca.png'
    plt.savefig('figures/pca.png')

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
