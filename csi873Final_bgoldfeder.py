# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 18:06:07 2017

@author: bruce

CSI 873 Fall 2017 Dr. Griva

Final Exam

"""
import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers
import matplotlib.pyplot as plt
style.use('ggplot')

# radial basis function where gamm = -(1/(2*sigma**2))
def rbf(x, y, gamma):
    return np.exp(-1 * gamma * linalg.norm(x-y)**2 )

class SVM:
    def __init__(self, kernel=rbf, C=None):
        self.kernel = kernel
        self.C = C
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        P = cvxopt.matrix(np.outer(y,y) * K)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1,n_samples))
        b = cvxopt.matrix(0.0)
        
        tmp1 = np.diag(np.ones(n_samples) * -1)
        tmp2 = np.identity(n_samples)
        G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
        tmp1 = np.zeros(n_samples)
        tmp2 = np.ones(n_samples) * self.C
        h = cvxopt.matrix(np.hstack((tmp1, tmp2)))



#TODO

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification


    
    #todo this may or may not need to be adapted
    def visualize(self):
        plt.show()
    
    
#todo This needs to be adapted to our 784 byte vectors and the binary
# Classifier e.g. 3's and 6's are 1 and -1 respectively
data_dict = {-1:np.array([[1,7],
                          [2,8],
                          [3,8],]),
             
             1:np.array([[5,1],
                         [6,-1],
                         [7,3],])}    
    
svm = SVM()
svm.fit(data=data_dict)

predict_us = testing_data #data to be predicted/tested?

for p in predict_us:
    svm.predict(p)

svm.visualize()