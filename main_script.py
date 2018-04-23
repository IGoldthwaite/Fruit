'''
Main script to load and run data on our classifiers.
'''

import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from classifiers import KMeansClassifier
from data_helper import load_data
from data_helper import print_accuracy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

def main_random_forest_test():    
    train_data, train_labels = load_data(True, -1)
    test_data, test_labels = load_data(False, -1)
    forest = RandomForestClassifier(n_estimators=10)
    forest = forest.fit(train_data, train_labels)
    predictions = forest.predict(test_data)
    print_accuracy(predictions, test_labels)

def main_combination_test():

    num_classes = 5
    train_data, train_labels = load_data(True, num_classes)
    test_data, test_labels = load_data(False, num_classes)
    
    pcadecomp = PCA(n_components=5)
    pca_train = pcadecomp.fit_transform(train_data)
    pca_test = pcadecomp.transform(test_data)

    label_to_val = {}
    val_to_label = {}
    ix = 0
    for cl in list(set(train_labels)):
        label_to_val[cl] = ix
        val_to_label[ix] = cl
        ix += 1

    train_values = np.array([label_to_val[s] for s in train_labels])

    kmclf = KMeansClassifier(num_classes*5)
    kmclf.train(pca_train, train_values)
    p_vals = kmclf.predict(pca_test)
    predictions = [val_to_label[p] for p in p_vals]
    
    print_accuracy(predictions, test_labels)

def main_kmeans_test():

    num_classes = 60
    train_data, train_labels = load_data(True, num_classes)
    
    # make things a bit easier for us
    label_to_val = {}
    val_to_label = {}
    ix = 0
    for cl in list(set(train_labels)):
        label_to_val[cl] = ix
        val_to_label[ix] = cl
        ix += 1

    # map class string labels to values 0 to num_classes-1
    train_values = np.array([label_to_val[s] for s in train_labels])

    kmclf = KMeansClassifier(num_classes*2)
    kmclf.train(train_data, train_values)
    
    test_data, test_labels = load_data(False, num_classes)
    p_vals = kmclf.predict(test_data)
    predictions = [val_to_label[p] for p in p_vals]

    print_accuracy(predictions, test_labels)


def main_pcatest():

    test_data, test_labels = load_data(False, -1)

    pcadecomp = PCA(n_components=3)
    pca_data = pcadecomp.fit_transform(test_data)
    pca_data = np.array(pca_data)

    classes = list(set(test_labels))
    pca_by_class = {} 
    for cl in classes:
        pca_by_class[cl] = np.array([pca_data[i] for i in range(len(pca_data)) if test_labels[i] == cl])
    
    fig = plt.figure()
    ax = Axes3D(fig)
    for cl in pca_by_class:
        ax.scatter(pca_by_class[cl].T[0], pca_by_class[cl].T[1], pca_by_class[cl].T[2]) 

    plt.savefig('figures/pca.png')

def main():
    #train_data, train_labels = load_data(True)
    test_data, test_labels = load_data(False)
    
    print test_data
    print test_labels
    print len(test_data)
    print len(test_data[0])

    img = test_data[0].reshape((45, 45, 3))
    print test_labels[0]

    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
#    main()
#    main_pcatest()
#    main_kmeans_test()
#    main_combination_test()
    main_random_forest_test()
