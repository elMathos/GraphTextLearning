"""
Compute micro and macro-averages precision and recall
"""

from sklearn.metrics import precision_score, recall_score


def precision(labels_true, labels_pred, avg_type):
    # avg_type is 'micro' or 'macro'
    return precision_score(labels_true, labels_pred, average=avg_type)


def recall(labels_true, labels_pred, avg_type):
    # avg_type is 'micro' or 'macro'
    return recall_score(labels_true, labels_pred, average=avg_type)
