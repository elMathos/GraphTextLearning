"""
Main file for the ALTeGraD final assignment

@authors: Mathurin Massias, Clement Nicolle, Michael Weiss
"""


from loaddata import load_data
from bagofwords import dictionary, tf_idf
from nearestneighbors import kNN_predict
from classif import SVM_predict, RF_predict, Adaboost_predict
from evaluation import *
import numpy as np
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
print "Micro averaging precision for NN: " + str(errorPercentage("kNN", labels_pred_kNN, test_labels))
print "Macro averaging precision for NN: " + str(recall(test_labels, labels_pred_kNN, "macro"))
print "Micro averaging recall for NN: " + str(recall(test_labels, labels_pred_kNN, "micro"))
print "Macro averaging recall for NN: " + str(precision(test_labels, labels_pred_kNN, "macro"))

# SVM
labels_pred_SVM = SVM_predict(tfidf_train, train_labels, tfidf_test)
print "Micro averaging precision for SVM: " + str(errorPercentage("SVM", labels_pred_SVM, test_labels))
print "Macro averaging precision for SVM: " + str(recall(test_labels, labels_pred_SVM, "macro"))
print "Micro averaging recall for SVM: " + str(recall(test_labels, labels_pred_SVM, "micro"))
print "Macro averaging recall for SVM: " + str(precision(test_labels, labels_pred_SVM, "macro"))

# Random Forest
precMicro = np.zeros(10)
recMacro = np.zeros(10)
recMicro = np.zeros(10)
precMacro = np.zeros(10)
for i in range(10):
    labels_pred_RF = RF_predict(tfidf_train, train_labels, tfidf_test)
    precMicro[i] = errorPercentage("RF", labels_pred_RF, test_labels)
    recMacro[i] = recall(test_labels, labels_pred_RF, "macro")
    recMicro[i] = recall(test_labels, labels_pred_RF, "micro")
    precMacro[i] = precision(test_labels, labels_pred_RF, "macro")
print "Micro averaging precision for RF: " + str(np.mean(precMicro))
print "Macro averaging precision for RF: " + str(np.mean(precMacro))
print "Micro averaging recall for RF: " + str(np.mean(recMicro))
print "Macro averaging recall for RF: " + str(np.mean(recMacro))

# Adaboost
labels_pred_Adaboost = Adaboost_predict(tfidf_train, train_labels, tfidf_test)
print "Micro averaging precision for RF: " + str(errorPercentage("Adaboost", labels_pred_Adaboost, test_labels))
print "Macro averaging precision for RF: " + str(recall(test_labels, labels_pred_Adaboost, "macro"))
print "Micro averaging recall for RF: " + str(recall(test_labels, labels_pred_Adaboost, "micro"))
print "Macro averaging recall for RF: " + str(precision(test_labels, labels_pred_Adaboost, "macro"))



# Cross val RF:
n_est = [1, 5, 10, 15, 20, 25, 30, 40, 50, 100, 200, 300, 400, 500]
err = np.zeros(len(n_est))
i = 0
for n in n_est:
    print i
    labels_pred_RF = RF_predict(tfidf_train, train_labels, tfidf_test, n_estim=n_est)
    err[i] = errorPercentage("RF", labels_pred_RF, test_labels)
    i  += 1
    print(i)
    