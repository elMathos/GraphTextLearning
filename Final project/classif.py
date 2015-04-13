"""
Bag-of-words classifiers
"""

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


def SVM_predict(train_data, train_labels, test_data,
                C=1.0, gamma=0.0, verbose=2):
    clf = svm.SVC(C=C, gamma=gamma)
    clf.fit(train_data, train_labels)
    return clf.predict(test_data)


def RF_predict(train_data, train_labels, test_data, n_estim=10, verbose=2):
    clf = RandomForestClassifier(n_estimators=n_estim)
    clf.fit(train_data, train_labels)
    return clf.predict(test_data)
