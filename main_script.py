'''
Main script to load and run data on our classifiers.
'''

import cv2
import numpy as np

from classifiers import KMeansClassifier
from data_helper import load_data
from data_helper import print_accuracy
from data_helper import apply_pca
from data_helper import plot_pca

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

def classify_svm(train_data, train_labels, test_data, test_labels, using_pca = False):
    print 'Training SVM' + (' with PCA' if using_pca else '')
    svm_classifier = svm.SVC()
    svm_classifier = svm_classifier.fit(train_data, train_labels)
    print 'Testing SVM' + (' with PCA' if using_pca else '')
    predictions = svm_classifier.predict(test_data)
    print_accuracy(predictions, test_labels)

def classify_naive_bayes(train_data, train_labels, test_data, test_labels, using_pca = False):    
    print 'Training Naive Bayes' + (' with PCA' if using_pca else '')
    nb = GaussianNB().fit(train_data, train_labels)
    print 'Testing Naive Bayes' + (' with PCA' if using_pca else '')
    predictions = nb.predict(test_data)
    print_accuracy(predictions, test_labels)

def classify_knn(train_data, train_labels, test_data, test_labels, k, using_pca = False):    
    print 'Training {}-NN'.format(k) + (' with PCA' if using_pca else '')
    knn = KNeighborsClassifier(n_neighbors = k).fit(train_data, train_labels)
    print 'Testing {}-NN'.format(k) + (' with PCA' if using_pca else '')
    predictions = knn.predict(test_data)
    print_accuracy(predictions, test_labels)

def classify_decision_tree(train_data, train_labels, test_data, test_labels, depth, using_pca = False):    
    print 'Training Decision Tree ({} max_depth)'.format(depth) + (' with PCA' if using_pca else '')
    tree = DecisionTreeClassifier(max_depth = depth).fit(train_data, train_labels)
    print 'Testing Decision Tree ({} max_depth)'.format(depth) + (' with PCA' if using_pca else '')
    predictions = tree.predict(test_data)
    print_accuracy(predictions, test_labels)

def classify_random_forest(train_data, train_labels, test_data, test_labels, estimators, using_pca = False):    
    print 'Training Random Forest ({} estimators)'.format(estimators) + (' with PCA' if using_pca else '')
    forest = RandomForestClassifier(n_estimators=estimators)
    forest = forest.fit(train_data, train_labels)
    print 'Testing Random Forest ({} estimators)'.format(estimators) + (' with PCA' if using_pca else '')
    predictions = forest.predict(test_data)
    print_accuracy(predictions, test_labels)

def classify_kmeans(train_data, train_labels, test_data, test_labels, num_classes, using_pca = False):
    print 'Training KMeans' + (' with PCA' if using_pca else '')
    
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
    
    print 'Testing KMeans' + (' with PCA' if using_pca else '')
    p_vals = kmclf.predict(test_data)
    predictions = [val_to_label[p] for p in p_vals]

    print_accuracy(predictions, test_labels)

def main():
    # Load data once
    train_data, train_labels = load_data(True)
    test_data, test_labels = load_data(False)

    # Plot PCA in 3 dimensions
    #plot_pca(test_data, test_labels)

    # Apply PCA to data
    train_data_pca, test_data_pca = apply_pca(train_data, test_data, 50)

    # Start classifying
    #for i in [2, 5, 10, 25, 50, 100]:
    #    classify_random_forest(train_data_pca, train_labels, test_data_pca, test_labels, i, True)
    #    classify_random_forest(train_data, train_labels, test_data, test_labels, i)

    #for i in [2, 5, 10, 25, 50, 100]:
    #    classify_decision_tree(train_data_pca, train_labels, test_data_pca, test_labels, i, True)
    #    classify_decision_tree(train_data, train_labels, test_data, test_labels, i)

    for i in [3, 5, 7]:
        classify_knn(train_data_pca, train_labels, test_data_pca, test_labels, i, True)
        classify_knn(train_data, train_labels, test_data, test_labels, i)
    #classify_naive_bayes(train_data, train_labels, test_data, test_labels)
    #classify_naive_bayes(train_data_pca, train_labels, test_data_pca, test_labels, True)
    #classify_svm(train_data_pca, train_labels, test_data_pca, test_labels, True)
    #classify_svm(train_data, train_labels, test_data, test_labels)
    #classify_kmeans(train_data, train_labels, test_data, test_labels, 60)
    #classify_kmeans(train_data_pca, train_labels, test_data_pca, test_labels, 60, True)

if __name__ == '__main__':
    main()
