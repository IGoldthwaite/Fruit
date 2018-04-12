'''
Main script to load and run data on our classifiers.
'''

import numpy as np
from data_helper import load_data

def main():
    train_data, train_labels = load_data(True)
    test_data, test_labels = load_data(False)

if __name__ == '__main__':
    main()
