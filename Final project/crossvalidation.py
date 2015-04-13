"""
Cross-validation on train dataset to find optimal values for parameters
"""

import numpy as np
from nearestneighbors import kNN_predict
from classif import SVM_predict, RF_predict
from evaluation import precision, recall


def crossval_kNN(n_fold, train_data, train_labels, k_range):
    n_k = len(k_range)
    micro_prec = np.zeros(n_k)
    macro_prec = np.zeros(n_k)
    macro_rec = np.zeros(n_k)
    subset_size = len(train_data)/n_fold
    for i in range(n_k):
        k = k_range[i]
        for j in range(n_fold):
            round_test_data = train_data[j*subset_size:(j+1)*subset_size]
            round_train_data = np.vstack((train_data[:j*subset_size],
                                          train_data[(j+1)*subset_size:]))
            round_test_lab = train_labels[i*subset_size:(j+1)*subset_size]
            round_train_lab = np.concatenate((train_labels[:j*subset_size],
                                              train_labels[(j+1) *
                                              subset_size:]))
            lab_pred = kNN_predict(k, round_train_data, round_train_lab,
                                   round_test_data)
        micro_prec[i] = precision(round_test_lab, lab_pred,
                                  avg_type='micro')
        macro_prec[i] = precision(round_test_lab, lab_pred,
                                  avg_type='macro')
        macro_rec[i] = recall(round_test_lab, lab_pred,
                              avg_type='macro')

    return micro_prec, macro_prec, macro_rec


def crossval_SVM(n_fold, train_data, train_labels, C_range, gamma_range):
    n_C = len(C_range)
    n_gamma = len(gamma_range)
    micro_prec = np.zeros((n_C, n_gamma))
    macro_prec = np.zeros((n_C, n_gamma))
    macro_rec = np.zeros((n_C, n_gamma))
    subset_size = len(train_data)/n_fold
    for i in range(n_C):
        C = C_range[i]
        for j in range(n_gamma):
            gamma = gamma_range[j]
            for l in range(n_fold):
                round_test_data = train_data[l*subset_size:(l+1)*subset_size]
                round_train_data = np.vstack((train_data[:l*subset_size],
                                              train_data[(l+1)*subset_size:]))
                round_test_lab = train_labels[i*subset_size:(l+1)*subset_size]
                round_train_lab = np.concatenate((train_labels[:l*subset_size],
                                                  train_labels[(l+1) *
                                                  subset_size:]))
                lab_pred = SVM_predict(round_train_data, round_train_lab,
                                       round_test_data, C=C, gamma=gamma)
            micro_prec[i, j] = precision(round_test_lab, lab_pred,
                                         avg_type='micro')
            macro_prec[i, j] = precision(round_test_lab, lab_pred,
                                         avg_type='macro')
            macro_rec[i, j] = recall(round_test_lab, lab_pred,
                                     avg_type='macro')

    return micro_prec, macro_prec, macro_rec


def crossval_RF(n_fold, train_data, train_labels, ntrees_range):
    n_ntrees = len(ntrees_range)
    micro_prec = np.zeros(n_ntrees)
    macro_prec = np.zeros(n_ntrees)
    macro_rec = np.zeros(n_ntrees)
    subset_size = len(train_data)/n_fold
    for i in range(n_ntrees):
        ntrees = ntrees_range[i]
        for j in range(n_fold):
            round_test_data = train_data[j*subset_size:(j+1)*subset_size]
            round_train_data = np.vstack((train_data[:j*subset_size],
                                          train_data[(j+1)*subset_size:]))
            round_test_lab = train_labels[i*subset_size:(j+1)*subset_size]
            round_train_lab = np.concatenate((train_labels[:j*subset_size],
                                              train_labels[(j+1) *
                                              subset_size:]))
            lab_pred = RF_predict(round_train_data, round_train_lab,
                                   round_test_data, n_estim = ntrees)
        micro_prec[i] = precision(round_test_lab, lab_pred,
                                  avg_type='micro')
        macro_prec[i] = precision(round_test_lab, lab_pred,
                                  avg_type='macro')
        macro_rec[i] = recall(round_test_lab, lab_pred,
                              avg_type='macro')

    return micro_prec, macro_prec, macro_rec
