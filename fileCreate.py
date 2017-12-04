# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:41:49 2017

@author: Bruce
"""

import os
import numpy as np

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
    
def ReadInOneList(fullData,maxRows,tOt):
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
            
    start = 0
    for out in range(0,(numRows*10)-1,maxRows):
        outpath = os.getcwd() + "\\data4\\" + tOt + str(out/maxRows) + ".txt"
        np.savetxt(outpath,np.asarray(allData[start:start+maxRows]))
        start += maxRows

def Driver(dpath,trnNum=250,tstNum=250):
    
    # Read in the Training data first
    datasetTrn = ReadInFiles(dpath,'test')
    my_data = ReadInOneList(datasetTrn,trnNum,'test')
    
Driver(os.getcwd()+'\data')