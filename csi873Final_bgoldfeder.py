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

class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r',-1:'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    # train
    def fit(self, data):
        self.data = data
        # { ||w||: [w,b] }
        opt_dict = {}

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]


#TODO

    def predict(self,features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification !=0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification

    # radial basis function where gamm = -(1/(2*sigma**2))
    def rbf(x, y, gamma):
        return np.exp(-1 * gamma * linalg.norm(x-y)**2 )
    
    
    
svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = #data to be predicted/tested?

for p in predict_us:
    svm.predict(p)

svm.visualize()