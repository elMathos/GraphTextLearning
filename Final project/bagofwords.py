"""
Represent the collection as a document-term matrix
"""

from __future__ import division
import numpy as np
from math import log


def dictionary(data):
    # extract the different terms
    dic = np.array([])
    for row in data:
        dic = np.hstack((dic, np.unique(row)))
    dic = np.unique(dic)
    dic = sorted(dic)
    return dic


def wordcount_matrix(data, dico):
    n_doc = len(data)
    n_words = len(dico)
    mat = np.zeros((n_doc, n_words))
    for i in range(n_doc):
        doc = np.asarray(data[i])
        print "Counting words for document %d over %d" % (i+1, n_doc)
        for j in range(n_words):
            word = dico[j]
            # nb of occurences of word in doc :
            mat[i, j] = len(np.where(doc == word)[0])

    return mat


def tf_idf(data, dico):
    mat = wordcount_matrix(data, dico)
    # from the wordcount matrix, compute the tf-idf weights
    n_doc = mat.shape[0]
    n_words = mat.shape[1]
    for j in range(n_words):
        df = len([i for i in range(n_doc) if mat[i, j] != 0])
        if df != 0:
            idf = log(n_doc/df)
        else:
            idf = 0
        # multiply tf in column j by idf
        mat[:, j] *= idf

    return mat
