"""
Nearest neighbors algorithm in vector space (cosine similarity)
"""

import numpy as np


def kNN_indices(k, tfidf_mat, query):
    # retrieve the row indices in tfidf_mat of the kNN from query
    # query is the tf_idf vector for test document
    n_doc = tfidf_mat.shape(0)
    similarities = np.zeros(n_doc)
    for i in range(n_doc):
        # cosine between query and doc no i
        similarities[i] = (np.dot(query, tfidf_mat[i]) /
                           (np.linalg.norm(query, 2) *
                            np.linalg.norm(tfidf_mat[i], 2)))
    # get the indices of the k maximum
    max_indices = similarities.argsort()[-k:][::-1]
    return max_indices


def most_common(L):
    return max(set(L), key=L.count)


def kNN_predict_onedoc(k, tfidf_mat, train_labels, query):
    max_indices = kNN_indices(k, tfidf_mat, query)
    corresponding_labels = sorted(train_labels[max_indices])
    # find the most common label
    # in case of equality, pick the first in alaphabetical order
    return most_common(corresponding_labels)


def kNN_predict(k, tfidf_train, train_labels, tfidf_test):
    pred = []
    n_test = tfidf_test.shape[0]
    for i in range(n_test):
        pred.append(kNN_predict_onedoc(k, tfidf_train,
                                       train_labels, tfidf_test[i]))
    return pred
