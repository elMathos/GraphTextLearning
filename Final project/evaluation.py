"""
Compute micro and macro-averages precision and recall
"""

from __future__ import division
from sklearn.metrics import precision_score, recall_score
import numpy as np

def precision(labels_true, labels_pred, avg_type):
    # avg_type is 'micro' or 'macro'
    return precision_score(labels_true, labels_pred, average=avg_type)


def recall(labels_true, labels_pred, avg_type):
    # avg_type is 'micro' or 'macro'
    return recall_score(labels_true, labels_pred, average=avg_type)


def errorPercentage(algo, labels_true, labels_pred):
    matching = np.array(labels_true) == np.array(labels_pred)
    perc = sum(matching)/len(matching)
    print str(100*perc) +  " % matching percentage for " + algo
    return perc