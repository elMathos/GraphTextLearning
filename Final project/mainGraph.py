# -*- coding: utf-8 -*-

"""
Main file for the ALTeGraD final assignment

@authors: Mathurin Massias, Clement Nicolle, Michael Weiss
"""

import networkx as nx
from loaddata import load_data
from bagofwords import dictionary
from graphofwordsweighted import GraphOfWordsWeighted
from nearestneighbors import kNN_predict
from classif import SVM_predict, RF_predict, Adaboost_predict

# BAG OF WORDS MODEL

# load the data and labels
train_data, train_labels, test_data, test_labels = load_data()

# get tf_idf matrices
dico = dictionary(train_data)


graphsTrain=GraphOfWordsWeighted(train_data,dico,4)
graphsTrain.compute_documentTerm(0,dico)
graphsTrain.penalize_idf(dico)
graphsTrain.normalize()

graphsTest=GraphOfWordsWeighted(test_data,dico,4)
graphsTest.compute_documentTerm(0,dico)
graphsTest.penalize_idf(dico)
graphsTest.normalize()


train=graphsTrain.documentTerm
# Nearest Neighbors
k = 5
labels_pred_kNN = kNN_predict(k, graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
# SVM
labels_pred_SVM = SVM_predict(graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
# Random Forest
labels_pred_RF = RF_predict(graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)
# Adaboost
labels_pred_Adaboost = Adaboost_predict(graphsTrain.documentTerm, train_labels, graphsTest.documentTerm)


## t
