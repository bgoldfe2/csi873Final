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
import os

def ReadInFiles(path,trnORtst):
    # This reads in all the files from a directory filtering on what the file
    # starts with
    fullData = []
    fnames = os.listdir(path)
    for fname in fnames:
        if fname.startswith(trnORtst):
            print (fname)
            data = np.loadtxt(path + "\\" + fname)
            fullData.append(data)
    #numFiles = len (fullData)
    #print(numFiles)
   
    return fullData
    
def ReadInOneList(fullData,maxRows):
    # This function combines all of the data into one array for ease of use
    # It contains a capping ability to configure how many results to use
    allData = []
    numFiles = len (fullData)
    for j in range (numFiles):
        # allows for smaller data set sizes
        numRows = len (fullData[j])
        #print('numrows,maxrows ',numRows,maxRows)
        if (maxRows < numRows):
            numRows = maxRows
    
        for k in range(numRows):
            allData.append(fullData[j][k])
    return np.asarray(allData)

def getData(dpath,trnNum,tstNum):

    # Read in the Training data first
    datasetTrn = ReadInFiles(dpath,'train')
    my_data = ReadInOneList(datasetTrn,trnNum)
    
    # Convert the 0-255 to 0 through 1 values in data
    my_data[:,1:] /= 255.0
    
    
    # randomize the rows for better training
    #np.random.shuffle(my_data)
    inNum,cols = my_data.shape    
    just_trn_data = my_data[:,1:]
    answerTrn = my_data[:,0]
    
    # Read in the test data
    #dpath2 = os.getcwd()+'\data3'
    dataset2 = ReadInFiles(dpath,'test')
    my_test = ReadInOneList(dataset2,tstNum) 
    
    tstNum,cols = my_test.shape
    #print('num rows ',tstNum)
    
    # Convert the 0-255 to 0 through 1 values in data
    my_test[:,1:] /= 255.0
    
    just_test_data = my_test[:,1:]
    answerTest = my_test[:,0] 
    
    return just_trn_data,answerTrn,just_test_data,answerTest

def reduceDataQuality(dsize,just_trn_data,just_test_data):   
    # 50% Reduced pixel data and label sets
    fiftyPtrnData = np.delete(just_trn_data, list(range(0, just_trn_data.shape[1], 2)), axis=1)
    fiftyPtstData = np.delete(just_test_data, list(range(0, just_test_data.shape[1], 2)), axis=1)
    
    # 75% Reduced pixel data and label sets
    seventyfivePtrnData = np.delete(fiftyPtrnData, list(range(0, fiftyPtrnData.shape[1], 2)), axis=1)
    seventyfivePtstData = np.delete(fiftyPtstData, list(range(0, fiftyPtstData.shape[1], 2)), axis=1)

    # 90% Reduced pixel data and label sets
    ninetyPtrnData = just_trn_data[:,::10].copy()
    ninetyPtstData = just_test_data[:,::10].copy()

    # 95% Reduced pixel data and label sets
    ninetyfivePtrnData = ninetyPtrnData[:,::2].copy()
    ninetyfivePtstData = ninetyPtstData[:,::2].copy()
    
    if dsize == 50:
        just_trn_data = fiftyPtrnData
        just_test_data = fiftyPtstData
    elif dsize == 75:
        just_trn_data = seventyfivePtrnData
        just_test_data = seventyfivePtstData
    elif dsize == 90:
        just_trn_data = ninetyPtrnData
        just_test_data = ninetyPtstData
    elif dsize == 95:
        just_trn_data = ninetyfivePtrnData
        just_test_data = ninetyfivePtstData
        
    return just_trn_data,just_test_data
    

def HeatMap(numberIn):
    #heat map to show numbers
    plt.matshow(numberIn.reshape(28,28))
    plt.colorbar()
    plt.show()

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

# radial basis function where gamm = -(1/(2*sigma**2))
def rbf(x, y, gamma):
    return np.exp(-1 * gamma * linalg.norm(x-y)**2 )

class SVM(object):
    def __init__(self, trnData, trnAns, tstData, tstAns,kernel=rbf, \
                 C=None, gamma=.05, trnNum=250, tstNum=250):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        if self.C is not None: self.C = float(self.C)
        self.trnNum = trnNum
        self.tstNum = trnNum
        self.trnData = trnData
        self.trnAns = trnAns
        self.tstData = tstData
        self.tstAns = tstAns     
        

    


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
        cvxopt.solvers.options['show_progress'] = False
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        #print("alphas? ",a[a>1])

        # Support vectors have non zero lagrange multipliers
        sv = a > 1e-1
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
            #print("for i_0 = ",ind[n]," and alpha_i0 = ",self.a[n]," b is ",each_b)
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
        #print ("project return value is ",y_predict + self.b)
        return y_predict + self.b

    # Checks the sign of the projection in order to determine which side of 
    # the decision boundary it is on
    def predict(self, X):
        return np.sign(self.project(X))
    
#todo This needs to be adapted to our 784 byte vectors and the binary
# Classifier e.g. 3's and 6's are 1 and -1 respectively

if __name__ == "__main__":
    
    
    def test_3v6(dset,trnData, trnAns, tstData, tstAns,trnN, tstN):
        
        # Adjust the data quality
        if dset != 100:
            trnData, tstData = reduceDataQuality(dset,trnData,tstData)
       
        # Test1 is the dual soft margin SVM to classify 3s vs 6s only
        # Get the training data for 250 3s and 250 6s
        test1 = SVM(trnData, trnAns, tstData, tstAns, \
                    kernel=rbf,C=100,gamma=.05, trnNum=trnN, tstNum=tstN)
        X_Train_3s = test1.trnData[test1.trnNum*3:(test1.trnNum*4)]
        y_Train_3s = test1.trnAns[test1.trnNum*3:(test1.trnNum*4)]
        
        X_Train_6s = test1.trnData[test1.trnNum*6:(test1.trnNum*7)]
        y_Train_6s = test1.trnAns[test1.trnNum*6:(test1.trnNum*7)]
                
        #HeatMap(X_Train_3s[0])
        #HeatMap(X_Train_3s[test1.trnNum-1])
        #HeatMap(X_Train_6s[0])
        #HeatMap(X_Train_6s[test1.trnNum-1])

        #print("first 3 ",y_Train_3s[0], " last 3 ",y_Train_3s[test1.trnNum-1])
        #print("first 6 ",y_Train_6s[0], " last 6 ",y_Train_6s[test1.trnNum-1])
        np.savetxt("y_Train_3s.txt",y_Train_3s)
        np.savetxt("y_Train_6s.txt",y_Train_6s)
        # Get the test data for 250 3s and 250 6s
        X_Test_3s = test1.tstData[test1.tstNum*3:(test1.tstNum*4)]
        y_Test_3s = test1.tstAns[test1.tstNum*3:(test1.tstNum*4)]
        
        X_Test_6s = test1.tstData[test1.tstNum*6:(test1.tstNum*7)]
        y_Test_6s = test1.tstAns[test1.tstNum*6:(test1.tstNum*7)]
    
    
        #HeatMap(X_Test_3s[0])
        #HeatMap(X_Test_3s[test1.tstNum-1])
        #HeatMap(X_Test_6s[0])
        #HeatMap(X_Test_6s[test1.tstNum-1])

        #print("first 3 ",y_Test_3s[0], " last 3 ",y_Test_3s[test1.tstNum-1])
        #print("first 6 ",y_Test_6s[0], " last 6 ",y_Test_6s[test1.tstNum-1])
      
    
        # The read in labels will be for data input checking only
        # I will convert the 3s labels to be -1 and
        # the 6s labels to be 1 for input into the SVM
        y_Train_3s = np.ones(test1.tstNum)
        y_Train_6s = np.ones(test1.tstNum) * -1
        
        X_train = np.vstack((X_Train_3s, X_Train_6s))
        y_train = np.hstack((y_Train_3s, y_Train_6s))
        
        y_Test_3s = np.ones(test1.tstNum)
        y_Test_6s = np.ones(test1.tstNum) * -1
        
        X_test = np.vstack((X_Test_3s, X_Test_6s))
        y_test = np.hstack((y_Test_3s, y_Test_6s))
        
        # Train the model using the full data set
        test1.fit(X_train, y_train)
        
        # Test model against the test data set
        y_predict = test1.predict(X_test)
        correct = np.sum(y_predict == y_test)
        print("Full data set %d out of %d predictions correct" % (correct, len(y_predict)))
        print("Full data set Accuracy of ",correct/len(y_predict))
        
    #Get the data destructed levels 50,75,90,95
    dpath = os.getcwd() + "\\data4\\"
    trnNum=250
    tstNum=250
    dsize =100
    trnData, trnAns, tstData, tstAns = getData(dpath,trnNum,tstNum)
    
###############  Reduced Pixel Quality Reduction  ################
        
    #test_3v6(100,trnData, trnAns, tstData, tstAns)
    #test_3v6(50,trnData, trnAns, tstData, tstAns)
    #test_3v6(75,trnData, trnAns, tstData, tstAns)
#    test_3v6(90,trnData, trnAns, tstData, tstAns,trnNum,tstNum)        
    #test_3v6(95,trnData, trnAns, tstData, tstAns)
    
################## SVD Reduced Quality Section #################

    U,D,Vt = linalg.svd(trnData,full_matrices=False)
    X_a = np.dot(np.dot(U, np.diag(D)), Vt)
    print(np.std(trnData), np.std(X_a), np.std(trnData - X_a))   
    print(D.shape)
    #np.savetxt("svdFull.txt",D)
    #np.savetxt("svd50.txt",D)
    percentVals = [392,196,78,39]
    for p in percentVals:
        D[p:]=0
        X_bar50 = np.dot(np.dot(U, np.diag(D)), Vt)
        print("SVD for",round(1-p/784,2),"% reduction")
        test_3v6(100,X_bar50, trnAns, tstData, tstAns,trnNum,tstNum)

    asdf    
################## Reduced Number of Samples Section
    trnData_50 = trnData[::2,:].copy()
    tstData_50 = tstData[::2,:].copy()
    trnAns_50 = trnAns[::2].copy()
    tstAns_50 = tstAns[::2].copy()
    trnNum = 125
    tstNum = 125
    #print("trn50 ",trnData_50.shape[0]," tst50 ",tstData_50.shape[0])

    test_3v6(100,trnData_50, trnAns_50, tstData_50, tstAns_50,trnNum,tstNum)
    
    # this will result in 62 and 63 (odd/even) outputs
    # to even out to 62 images per number, I remove every 125th
    remove_even_63s = [0,125,250,375,500]
    
    r75 = trnData_50[::2,:].copy()
    s75 = tstData_50[::2,:].copy()
    trnData_75 = np.delete(r75,remove_even_63s,axis=0)
    tstData_75 = np.delete(s75,remove_even_63s,axis=0)
    trnAns_75 = np.delete(trnAns_50[::2],remove_even_63s,axis=0)
    tstAns_75 = np.delete(tstAns_50[::2],remove_even_63s,axis=0)
    #np.savetxt("r75.txt",trnAns_75)
    #np.savetxt("trnData_75.txt",tstAns_75)
    
    trnNum = 62
    tstNum = 62
    #print("trn75 ",trnData_75.shape[0]," tst75 ",tstData_75.shape[0])
    #print(tstAns[62*3]," ",tstAns[62*4])
    
    test_3v6(100,trnData_75, trnAns_75, tstData_75, tstAns_75,trnNum,tstNum)

    # The 90% test for number of examples    
    trnData_90 = trnData[::10,:].copy()
    tstData_90 = tstData[::10,:].copy()
    
    trnAns_90 = trnAns[::10].copy()
    tstAns_90 = tstAns[::10].copy()
    #np.savetxt("trnData_90.txt",tstAns_90)
    
    trnNum = 25
    tstNum = 25
    
    test_3v6(100,trnData_90, trnAns_90, tstData_90, tstAns_90,trnNum,tstNum)
    
    # this will result in 12 and 13 (odd/even) outputs
    # to even out to 12 images per number, I remove every 25th
    remove_even_13s = [0,25,50,75,100]
    
    r95 = trnData_90[::2,:].copy()
    s95 = tstData_90[::2,:].copy()
    trnData_95 = np.delete(r95,remove_even_13s,axis=0)
    tstData_95 = np.delete(s95,remove_even_13s,axis=0)
    trnAns_95 = np.delete(trnAns_90[::2],remove_even_13s,axis=0)
    tstAns_95 = np.delete(tstAns_90[::2],remove_even_13s,axis=0)
    #np.savetxt("r95.txt",trnAns_95)
    #np.savetxt("trnData_95.txt",tstAns_95)
    
    trnNum = 12
    tstNum = 12
    #print("trn95 ",trnData_95.shape[0]," tst95 ",tstData_95.shape[0])
    
    test_3v6(100,trnData_95, trnAns_95, tstData_95, tstAns_95,trnNum,tstNum)
    
    