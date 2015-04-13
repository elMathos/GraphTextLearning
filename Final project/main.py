"""
Main file for the ALTeGraD final assignment

@authors: Mathurin Massias, Clement Nicolle, Michael Weiss
"""


from loaddata import load_data
from bagofwords import dictionary, tf_idf
from nearestneighbors import kNN_predict
from classif import SVM_predict, RF_predict, Adaboost_predict

# BAG OF WORDS MODEL

# load the data and labels
train_data, train_labels, test_data, test_labels = load_data()

# get tf_idf matrices
dico = dictionary(train_data)
tfidf_train = tf_idf(train_data, dico)
tfidf_test = tf_idf(test_data, dico)

# Nearest Neighbors
k = 5
labels_pred_kNN = kNN_predict(k, tfidf_train, train_labels, tfidf_test)
# SVM
labels_pred_SVM = SVM_predict(tfidf_train, train_labels, tfidf_test)
# Random Forest
labels_pred_RF = RF_predict(tfidf_train, train_labels, tfidf_test)
# Adaboost
labels_pred_Adaboost = Adaboost_predict(tfidf_train, train_labels, tfidf_test)
