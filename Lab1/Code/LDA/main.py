import math
import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from my_LDA import my_LDA
from predict import predict


K = 10000 #we repeat the error evaluation K times
errors = np.zeros(K)
my_data = np.genfromtxt('wine_data.csv', delimiter=',')


for k in range(K):
    np.random.shuffle(my_data) # shuffle datataset
    trainingData = my_data[:100,1:] # training data
    trainingLabels = my_data[:100,0] # class labels of training data
    
    testData = my_data[101:,1:] # test data
    testLabels = my_data[101:,0] # class labels of test data
    # Training LDA classifier
    W, projected_centroid, X_lda = my_LDA(trainingData, trainingLabels)
    
    # Perform predictions for the test data
    predictedLabels = predict(testData, projected_centroid, W)
    predictedLabels = predictedLabels+1
    
    
    # Compute accuracy
    counter = float(predictedLabels.size)
    for i in range(predictedLabels.size):
        if predictedLabels[i] == testLabels[i]:
            counter -= 1
    errors[k] = (counter / float(predictedLabels.size) * 100.0)
    
    
print "Mean relative classification error: ", np.mean(errors)
print "Max relative classification error: ", np.max(errors)
print "Min classification error: ", np.min(errors)
print "Std dev of the classification error", np.std(errors)

print "Mean number of classification errors: ", np.mean(errors*77/100)
print "Max number of classification errors: ", np.max(errors*77/100)
print "Min number of classification errors: ", np.min(errors*77/100)
print "Std number of classification errors", np.std(errors*77/100)



plt.figure(1)
plt.hist(errors*77/100, bins = 9)
plt.title("Histogram of classification error")
plt.show()


