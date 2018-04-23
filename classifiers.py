'''
Classifiers go here.
'''

import os, sys
import numpy as np
import random
from scipy import stats

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


class KMeansClassifier():
    '''
    Basic k-means classifier implementation.
    '''

    def __init__(self, k):
        self.k = k
        self.centers = None
        self.center_labels = None

    def train(self, X, Y):

        # initialize classifier variables
        num_features = len(X[0])
        pre_centers = -1*np.random.rand(self.k, num_features) 
        self.centers = 255*np.random.rand(self.k, num_features)
        self.center_labels = np.zeros(self.k, dtype=np.int)
        converged = False

        # send out clusters
        while not converged:

            # assign vectors and labels to centers
            kassignments = [[] for _ in self.centers]
            lassignments = [[] for _ in self.centers]
            for vi in range(len(X)):
                dists = [np.linalg.norm(X[vi]-c) for c in self.centers]
                kassignments[np.argmin(dists)].append(X[vi])
                lassignments[np.argmin(dists)].append(Y[vi])

            # if no points assigned, re-randomize those centers
            for cli in range(len(kassignments)):
                if len(kassignments[cli]) == 0:
                    self.centers[cli] = X[np.random.choice(range(len(X)))] 
                    #self.centers = 255*np.random.rand(self.k, num_features)
                    #break
                else:
                    self.centers[cli] = np.average(kassignments[cli], axis=0)

            if (pre_centers == self.centers).all():
                converged = True
            else:
                # re-cast here to copy array, not reference
                pre_centers = np.array(self.centers)

        # assign clusters labels
        self.center_labels = [stats.mode(x)[0][0] for x in lassignments]
        
    def predict(self, X, Y=None):

        predictions = []
        for vi in range(len(X)):
            dists = [np.linalg.norm(X[vi]-c) for c in self.centers]
            predictions.append(self.center_labels[np.argmin(dists)])

        return predictions


