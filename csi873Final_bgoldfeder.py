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

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

# radial basis function where gamm = -(1/(2*sigma**2))
def rbf(x, y, gamma):
    return np.exp(-1 * gamma * linalg.norm(x-y)**2 )

class SVM:
    def __init__(self, kernel=rbf, C=None, gamma=.02):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        if self.C is not None: self.C = float(self.C)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Gram matrix For RBF using gamma
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j],self.gamma)

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

        # solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        print("alphas? ",a[a>1])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-2
        #sv = (a > 1e-2) & (self.C - a > 1e-2)
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.a), n_samples))

        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            inside_sum = self.a * self.sv_y * K[ind[n],sv]
            each_b = self.sv_y[n] - np.sum(inside_sum,axis=0)
            print("for i_0 = ",ind[n]," and alpha_i0 = ",self.a[n]," b is ",each_b)
            #self.b -= np.sum(self.a * self.sv_y * K[ind[n],sv])
            self.b -= np.sum(inside_sum)

        self.b /= len(self.a)
        print("b is ",self.b)

    def project(self, X):
        # Calculates y_i * (w.x+b)
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.a, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv, self.gamma)
            y_predict[i] = s
        return y_predict + self.b

    # Checks the sign of the projection in order to determine which side of 
    # the decision boundary it is on
    def predict(self, X):
        return np.sign(self.project(X))

    
    #todo this may or may not need to be adapted
    def visualize(self):
        plt.show()
    
    
#todo This needs to be adapted to our 784 byte vectors and the binary
# Classifier e.g. 3's and 6's are 1 and -1 respectively

if __name__ == "__main__":
    import pylab as pl
    
    def gen_lin_separable_overlap_data():
        # generate training data in the 2-d case
        mean1 = np.array([0, 2])
        mean2 = np.array([2, 0])
        cov = np.array([[1.5, 1.0], [1.0, 1.5]])
        X1 = np.random.multivariate_normal(mean1, cov, 100)
        y1 = np.ones(len(X1))
        X2 = np.random.multivariate_normal(mean2, cov, 100)
        y2 = np.ones(len(X2)) * -1
        return X1, y1, X2, y2
    
    def split_train(X1, y1, X2, y2):
        X1_train = X1[:90]
        y1_train = y1[:90]
        X2_train = X2[:90]
        y2_train = y2[:90]
        X_train = np.vstack((X1_train, X2_train))
        y_train = np.hstack((y1_train, y2_train))
        return X_train, y_train

    def split_test(X1, y1, X2, y2):
        X1_test = X1[90:]
        y1_test = y1[90:]
        X2_test = X2[90:]
        y2_test = y2[90:]
        X_test = np.vstack((X1_test, X2_test))
        y_test = np.hstack((y1_test, y2_test))
        return X_test, y_test

    def test_soft2():
        X1, y1, X2, y2 = gen_lin_separable_overlap_data()
        X_train, y_train = split_train(X1, y1, X2, y2)
        X_test, y_test = split_test(X1, y1, X2, y2)
    
        clf = SVM(kernel=rbf,C=1000.1)
        clf.fit(X_train, y_train)
    
        y_predict = clf.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("%d out of %d predictions correct" % (correct, len(y_predict)))
    
        #plot_contour(X_train[y_train==1], X_train[y_train==-1], clf)
        
    test_soft2()