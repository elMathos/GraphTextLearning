from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

#================= ADD YOUR CODE HERE ====================================
# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
# Instructions: Perform SVD decomposition of matrix X. Save the 
#               three factors in variables U, S and V
U, S, V = linalg.svd(X)
U = matrix(U)
V = matrix(V)



#=========================================================================

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X20, X50, X100, X200 for each low rank approximation
## using the top k = [10, 20, 50, 100, 200] singlular values 
#
k_values = [10, 20, 50, 100, 200]
lowRankApprox = []
for i in range(5):
    k = k_values[i]
    lowRankApprox.append(U[:, :k] * diag(S[:k]) * V[:k, :])



#=========================================================================



#================= ADD YOUR CODE HERE ====================================
# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
# The Frobenius error can be computed as |X - X_k| / |X|
#
approxErrors = []
for i in range(5):
    k = k_values[i]
    approxErrors.append(linalg.norm(X-lowRankApprox[i])/linalg.norm(X))

plt.plot(k_values, approxErrors)


#=========================================================================
X10 = lowRankApprox[0]
X20 = lowRankApprox[1]
X50 = lowRankApprox[2]
X100 = lowRankApprox[3]
X200 = lowRankApprox[4]


# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X10,cmap = cm.Greys_r)
plt.title('Best rank' + str(10) + ' approximation')
plt.axis('off')


# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()


#================= ADD YOUR CODE HERE ====================================
# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k
plt.figure(3)
plt.plot(S)
plt.title("Singular values of X")


#=========================================================================

plt.show() 

plt.figure(4)
percentageEnergy = cumsum(S**2)/sum(S**2)
plt.plot(percentageEnergy)
relativeSValue = zeros(len(S) - 1)
for i in range(len(S) -1):
    relativeSValue[i] = log(S[i]/S[i+1])
plt.figure(5)
plt.plot(relativeSValue)


