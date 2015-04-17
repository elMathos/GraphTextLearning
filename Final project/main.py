"""
Main file for the ALTeGraD final assignment

@authors: Mathurin Massias, Clement Nicolle, Michael Weiss
"""


from loaddata import load_data
from bagofwords import dictionary, tf_idf
from nearestneighbors import kNN_predict
from classif import SVM_predict, RF_predict, Adaboost_predict
from evaluation import errorPercentage
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
errorPercentage("kNN", labels_pred_kNN, test_labels)
# SVM
labels_pred_SVM = SVM_predict(tfidf_train, train_labels, tfidf_test)
errorPercentage("SVM", labels_pred_SVM, test_labels)
# Random Forest
labels_pred_RF = RF_predict(tfidf_train, train_labels, tfidf_test)
errorPercentage("RF", labels_pred_RF, test_labels)
# Adaboost
labels_pred_Adaboost = Adaboost_predict(tfidf_train, train_labels, tfidf_test)
errorPercentage("Adaboost", labels_pred_Adaboost, test_labels)