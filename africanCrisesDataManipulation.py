from random import Random
import numpy as np
import pandas as pd
import DataManipulation
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

africadata = DataManipulation.DataManipulation()
def main():
    readData = pd.read_csv("./african_crises_raw_data.csv")
    readData['banking_crisis'] = africadata.replaceStrWithInt(readData, 'banking_crisis', "crisis", 1)
    readData['banking_crisis'] = africadata.replaceStrWithInt(readData, 'banking_crisis', "no_crisis", 0)
    readData = africadata.dropColumn(readData, 'case')
    readData = africadata.dropColumn(readData, 'cc3')
    readData = africadata.dropColumn(readData, 'country')  
    minVal = africadata.findMin(readData, 'year')
    readData['year'] = africadata.subtractScalar(readData, 'year', minVal)
    readData = africadata.clipData(readData, -500, 500)
    readData['exch_usd'] = africadata.standardizeColumn(readData, 'exch_usd')
    readData['inflation_annual_cpi'] = africadata.standardizeColumn(readData, 'inflation_annual_cpi')
    #swap systemic_crisis with banking_crisis
    readData = africadata.swapColumns(readData, 'year', 'systemic_crisis')
    #slice out X matrix
    X = readData.iloc[0:readData.shape[0], 1:readData.shape[1]].to_numpy()
    #slice out y vector
    y = readData.iloc[0:readData.shape[0], 0].to_numpy()
    w = np.ones((readData.shape[1] - 1))
    '''someTree = runDecisionTree(X, y)
    print(someTree.predict([[10,-.4,0,1,1,-0.3,0,0,0,1]]))'''
    '''someForest = runRandomForest(X,y)
    print(someForest.predict([[10,70,1,1,1,9,1,1,1,0]]))'''
    #w = trainPerceptron(X, y, w, .01, 200)
    #print(w)
    #print(testPerceptron(X, y,w))
    #africadata.writeToCsv(readData, './adjusted_african_crises_data.csv')

def testPerceptron(X, y, w):
    correct = 0
    res = np.dot(X, w)
    for i in range(y.shape[0]):
        if res[i] <= 0 and y[i] <= 0:
            correct = correct + 1
        elif res[i] > 0 and y[i] > 0:
            correct = correct + 1  
    return correct/y.shape[0]

def trainPerceptron(X, y, w, tolerance, max_iter):
    temp = np.dot(X,w)
    itCount = 0
    while testPerceptron(X, y, w) < (1-tolerance) and itCount < max_iter:
        for i in range(y.shape[0]):
            if y[i]==1 and temp[i] <= 0:
                w = w + X[i]
            if y[i]==0 and temp[i] > 0:
                w = w - X[i]
        itCount = itCount + 1
    
    if itCount == max_iter:
        return np.zeros((X.shape[1]))
    else:
        return w

def runPerceptron(): 
    acPerceptron = Perceptron(
        max_iter = 200,
        tol = .01
    )
    acPerceptron.fit()
    
    
    
def runDecisionTree(X, y):
    acTree = tree.DecisionTreeClassifier()
    return acTree.fit(X, y)


def runRandomForest(X,y):
    acForest = RandomForestClassifier()
    return acForest.fit(X,y)  

if __name__ == '__main__':
    main()