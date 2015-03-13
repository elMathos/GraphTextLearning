import numpy as np
import scipy as sp
import scipy.linalg as linalg

def my_LDA(X, Y):
    """
    Train a LDA classifier from the training set
    X: training data
    Y: class labels of training data

    """    
    
    classLabels = np.unique(Y) # different class labels on the dataset
    classNum = len(classLabels) # number of classes
    datanum, dim = X.shape # dimensions of the dataset
    totalMean = np.mean(X,0) # total mean of the data

    # ====================== YOUR CODE HERE ======================
    # Instructions: Implement the LDA technique, following the
    # steps given in the pseudocode on the assignment.
    # The function should return the projection matrix W,
    # the centroid vector for each class projected to the new
    # space defined by W and the projected data X_lda.
    Sw = np.zeros([dim, dim])
    Sb = np.zeros([dim, dim])
    centroids = np.zeros([dim, classNum])
    for i in classLabels:
        cluster_i = X[Y == i]
        centroids[:, i-1] = np.mean(cluster_i, 0) #labels start at 1
        for x in cluster_i:        
            Sw += np.transpose(np.matrix(x-centroids[:, i-1]))*np.matrix(x-centroids[:, i-1])
        Sb += len(cluster_i)*np.transpose(np.matrix(centroids[:, i-1]-totalMean))*np.matrix(centroids[:, i-1]-totalMean)
    
    eigval, eigvec = np.linalg.eig(np.linalg.inv(Sw)*Sb)
    eigvec = np.matrix(eigvec)
    W = eigvec[:, :(classNum-1)]
    X_lda = X * W
    projected_centroid = np.transpose(centroids) * W
    

    return [W, projected_centroid, X_lda]