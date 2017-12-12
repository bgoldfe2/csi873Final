# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 19:25:07 2017

@author: Bruce
"""
import numpy as np

predictMax = np.loadtxt("max2.txt")
values = np.loadtxt("values.txt")
print("predictMax",predictMax.shape)
print("values",values.shape)
ds=100
vt = values.transpose()
out = []
for y in range(0,1000):
    out.append(np.argmax(values[:,y]))
    
np.savetxt("maxLatest.txt",out)

"""
for x in range(0,10):
    
    correct = np.sum(predictMax[x*ds:((x+1)*ds)] == x)
    print("The MAX number",str(x),"using all the hyperplanes max distance %d out of %d predictions correct" % (correct, ((x+1)*ds)-x*ds))
    print("The MAX number",str(x),"versus Rest Accuracy of ",correct/ds)
    
"""
