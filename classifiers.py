'''
Classifiers go here.
'''

import os, sys
import numpy as np
import random

DEBUG = 2

class SomeOtherClassifier():
    '''
    Throw our classifiers in this script with an outline similar to this.
    '''

    def __init__(self):
        pass

    def train(X, Y):
        return self

    def predict(X, Y=None):
        return [0 for i in X]

class LMSKernelClassifier():
    '''
    Attempt to duplicate the homework 2 matlab code I wrote.
    '''

    def __init__(self, kernel='Gaussian', bandwidths=[1e2], lambds=[1e-4]):
        self.kernel = kernel
        self.bandwidths = bandwidths
        self.ls = lambds
        self.train_data = None
        self.K = None
        self.A = None

    def train(self, X, Y):
        '''
        Train on the data in X and labels in Y.
        '''
        
        self.train_data = np.array(X)

        train_ixs = random.sample([i for i in range(len(self.train_data))], 0.5*len(self.train_data))
        test_ixs = [i for i in range(len(self.train_data)) if i not in train_ixs]

        # split into cross-validation train/test
        train_split = self.train_data[train_ixs]
        train_labels = np.array(Y)[train_ixs]
        test_split = self.train_data[test_ixs]
        test_labels = np.array(Y)[test_ixs]

        error_score = np.zeros(len(self.bandwidths) * len(self.ls))

        cv_count = 0
        for l in self.ls:
            for b in self.bandwidths:
                
                cv_count += 1
                if DEBUG > 1:
                    sys.stdout.write('\rcross-validating {} of {}... '.format(cv_count, len(error_score)))

                if self.kernel == 'Gaussian':

                    # calculate K matrix
                    self.K = np.zeros((len(train_split), len(train_split)))
                    for i in range(len(train_split)):
                        
                        if DEBUG > 1:
                            sys.stdout.write('{}'.format(i))
                            sys.stdout.flush()
                        
                        self.K[i,:] = np.exp(-1*(np.sqrt(np.sum(train_split[i,:] - train_split)**2)**2) / (b**2))

                    # find least-squares solution A
                    self.A = np.linalg.inv(self.K + l*np.identity(len(train_split)))*train_labels


                    # just realized this isn't really applicable to our dataset so I'm going to stop here lol.
