# -*- coding: utf-8 -*-

"""
Main file for the ALTeGraD final assignment

@authors: Mathurin Massias, Clement Nicolle, Michael Weiss
"""

from loaddata import load_data
import time
from bagofwords import dictionary
from graphofwords import GraphOfWords
from nearestneighbors import kNN_predict
from classif import SVM_predict, RF_predict, Adaboost_predict
from evaluation import errorPercentage
start =time.time()
# BAG OF WORDS MODEL

# load the data and labels
train_data, train_labels, test_data, test_labels = load_data()

# get tf_idf matrices
dico = dictionary(train_data)


graphsTrain=GraphOfWords(train_data,dico,4)
graphsTrain.compute_documentTerm(1,dico)
graphsTrain.penalize_idf(dico)
graphsTrain.normalize()

graphsTest=GraphOfWords(test_data,dico,4)
graphsTest.compute_documentTerm(1,dico)
graphsTest.penalize_idf(dico)
graphsTest.normalize()

timeStruc=time.time()-start
train=graphsTrain.documentTerm
# Nearest Neighbors
k = 5
labels_pred_kNN = kNN_predict(k, graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
errorPercentage("kNN", labels_pred_kNN, test_labels)

timeKNN=time.time()-timeStruc
# SVM
labels_pred_SVM = SVM_predict(graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
errorPercentage("SVM", labels_pred_SVM, test_labels)
timeSVM=time.time()-timeKNN
# Random Forest
labels_pred_RF = RF_predict(graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
errorPercentage("RF", labels_pred_RF, test_labels)
timeRF=time.time()-timeSVM
# Adaboost
labels_pred_Adaboost = Adaboost_predict(graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
errorPercentage("Adaboost", labels_pred_Adaboost, test_labels)
timeADA=time.time()-timeRF
