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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

def main_svm_test(train_data, train_labels, test_data, test_labels):
    print 'Training SVM'
    svm_classifier = svm.SVC()
    svm_classifier = svm_classifier.fit(train_data, train_labels)
    print 'Testing SVM' 
    predictions = svm_classifier.predict(test_data)
    print_accuracy(predictions, test_labels)

def main_naive_bayes_test(train_data, train_labels, test_data, test_labels):    
    print 'Training Naive Bayes'
    nb = GaussianNB().fit(train_data, train_labels)
    print 'Testing Naive Bayes'
    predictions = nb.predict(test_data)
    print_accuracy(predictions, test_labels)

def main_knn_test(train_data, train_labels, test_data, test_labels, k):    
    print 'Training {}-NN'.format(k)
    knn = KNeighborsClassifier(n_neighbors = k).fit(train_data, train_labels)
    print 'Testing {}-NN'.format(k)
    predictions = knn.predict(test_data)
    print_accuracy(predictions, test_labels)

def main_decision_tree_test(train_data, train_labels, test_data, test_labels, depth):    
    print 'Training Decision Tree'
    tree = DecisionTreeClassifier(max_depth = depth).fit(train_data, train_labels)
    print 'Testing Decision Tree'
    predictions = tree.predict(test_data)
    print_accuracy(predictions, test_labels)

def main_random_forest_test(train_data, train_labels, test_data, test_labels, estimators):    
    print 'Training Random Forest'
    forest = RandomForestClassifier(n_estimators=estimators)
    forest = forest.fit(train_data, train_labels)
    print 'Testing Random Forest'
    predictions = forest.predict(test_data)
    print_accuracy(predictions, test_labels)

def main_combination_test(train_data, train_labels, test_data, test_labels, dimensions, num_classes):
    print 'Training PCA/KMeans Combination'
    
    pcadecomp = PCA(n_components=dimensions)
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

    kmclf = KMeansClassifier(num_classes*2)
    kmclf.train(pca_train, train_values)

    print 'Testing PCA/KMeans Combination'
    p_vals = kmclf.predict(pca_test)
    predictions = [val_to_label[p] for p in p_vals]
    
    print_accuracy(predictions, test_labels)

def main_kmeans_test(train_data, train_labels, test_data, test_labels, num_classes):
    print 'Training KMeans'
    
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
    
    print 'Testing KMeans'
    p_vals = kmclf.predict(test_data)
    predictions = [val_to_label[p] for p in p_vals]

    print_accuracy(predictions, test_labels)


def main_pcatest(data, labels, dimensions):
    print 'Applying PCA'

    pcadecomp = PCA(n_components=dimensions)
    pca_data = pcadecomp.fit_transform(data)
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

def main():
    # Load data once
    train_data, train_labels = load_data(True)
    test_data, test_labels = load_data(False)

    # Run classifiers
    #main_pcatest(test_data, test_labels, 3)
    #main_kmeans_test(train_data, train_labels, test_data, test_labels, 60)
    #main_combination_test(train_data, train_labels, test_data, test_labels, 5, 60)
    main_random_forest_test(train_data, train_labels, test_data, test_labels, 10)
    #main_decision_tree_test(train_data, train_labels, test_data, test_labels, 2)
    #main_knn_test(train_data, train_labels, test_data, test_labels, 3)
    #main_naive_bayes_test(train_data, train_labels, test_data, test_labels)
    #main_svm_test(train_data, train_labels, test_data, test_labels)

if __name__ == '__main__':
    main()
