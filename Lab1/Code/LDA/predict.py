import numpy as np
import scipy as sp
import scipy.linalg as linalg


def predict(X, projected_centroid, W):
    
    """Apply the trained LDA classifier on the test data 
    X: test data
    projected_centroid: centroid vectors of each class projected to the new space
    W: projection matrix computed by LDA
    """


    # Project test data onto the LDA space defined by W
    projected_data  = np.dot(X, W)
    
    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in the code to implement the classification
    # part of LDA. Follow the steps given in the assigment.
    classNum = projected_centroid.shape[0]
    label = np.zeros(projected_data.shape[0])
    for i in range(projected_data.shape[0]):
        norms = np.zeros(classNum)
        for j in range(classNum):
            norms[j] = np.linalg.norm(projected_data[i, :]-projected_centroid[j, :])
        label[i] = np.argmin(norms)
    
    
    
    # =============================================================

    # Return the predicted labels of the test data X
    return label