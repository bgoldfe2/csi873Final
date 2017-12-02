# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 14:27:12 2017

@author: Bruce
"""

import os
import mnist, svmcmpl, cvxopt, random

digits1 = [ 0 ]
digits2 = [ 1 ]

m1 = 4000; m2 = 4000

# read training data
dpath = os.getcwd()+"\\data\\mnist"
print( dpath)

images, labels = mnist.read(digits1 + digits2, dataset = "training", path = dpath )
images = images / 256.

C1 = [ k for k in range(len(labels)) if labels[k] in digits1 ]
C2 = [ k for k in range(len(labels)) if labels[k] in digits2 ]

random.seed()
random.shuffle(C1)
random.shuffle(C2)

train = C1[:m1] + C2[:m2]
random.shuffle(train)
X = images[train,:]
d = cvxopt.matrix([ 2*(k in digits1) - 1 for k in labels[train] ])

gamma = 4.0
sol = svmcmpl.softmargin_appr(X, d, gamma, width = 50, kernel = 'rbf', sigma = 2**5)