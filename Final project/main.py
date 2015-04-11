"""
Main file for the ALTeGraD final assignment

@authors: Mathurin Massias, Clement Nicolle, Michael Weiss
"""


from loaddata import load_data
from bagofwords import tf_idf
from nearestneighbors import kNN_predict

# load the data and labels
train_data, train_labels, test_data, test_labels = load_data()

# get tf_idf matrices
tfidf_train = tf_idf(train_data)
tfidf_test = tf_idf(test_data)

# Nearest Neighbors
k = 5
labels_pred_kNN = kNN_predict(k, tfidf_train, train_labels, tfidf_test)
