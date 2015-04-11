"""
Load data
"""

import csv


def load_data():
    # load train and test data and separate labels

    # train data
    train_datafile = open('Data/train.txt', 'r')
    train_reader = csv.reader(train_datafile)
    train_data = []
    for row in train_reader:
        train_data.append(row[0].split())
    train_labels = []
    for row in train_data:
        train_labels.append(row[0])
        row.pop(0)  # remove label from the row

    # test data
    test_datafile = open('Data/test.txt', 'r')
    test_reader = csv.reader(test_datafile)
    test_data = []
    for row in test_reader:
        test_data.append(row[0].split())
    test_labels = []
    for row in test_data:
        test_labels.append(row[0])
        row.pop(0)  # remove label from the row

    return train_data, train_labels, test_data, test_labels
