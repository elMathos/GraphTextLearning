"""
Bag-of-words classifiers
"""

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import time


def SVM_predict(train_data, train_labels, test_data,
                C=1.0, gamma=0.1, verbose=2):
    start = time.time()
    clf = svm.SVC(C=C, gamma=gamma)
    clf.fit(train_data, train_labels)
    print "Elapsed time: " + str(time.time() - start)
    return clf.predict(test_data)


def RF_predict(train_data, train_labels, test_data, n_estim=10, verbose=2):
    start = time.time()
    clf = RandomForestClassifier(n_estimators=n_estim)
    clf.fit(train_data, train_labels)
    print "Elapsed time: " + str(time.time() - start)
    return clf.predict(test_data)


def Adaboost_predict(train_data, train_labels, test_data,
                     n_estim=50, verbose=2):
    start = time.time()
    clf = AdaBoostClassifier(n_estimators=n_estim)
    clf.fit(train_data, train_labels)
    print "Elapsed time: " + str(time.time() - start)
    return clf.predict(test_data)
