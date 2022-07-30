from random import Random
import numpy as np
import pandas as pd
import DataManipulation
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

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
    readData = africadata.shuffleData(readData)
    #slice out X matrix
    X = readData.iloc[0:readData.shape[0], 1:readData.shape[1]].to_numpy()
    #slice out y vector
    y = readData.iloc[0:readData.shape[0], 0].to_numpy()
    #w = np.ones((readData.shape[1] - 1))
    
    #split into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    
    #run and test decision tree
    someTree = runDecisionTree(X_train, y_train) 
    params = {'min_samples_split': [2, 5, 10], 'min_samples_leaf': range(1,10), }
    clf = GridSearchCV(someTree, params)
    print("Decision Tree--------------------")
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train)
    print(scores)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    true_positive = np.sum(np.logical_and(y_pred, y_test))
    true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    false_positive = 0
    false_negative = 0
    for i in range(0, y_test.size):
        if y_pred[i] and not y_test[i]:
            false_positive = false_positive + 1
        elif y_test[i] and not y_pred[i]:
            false_negative = false_negative + 1
    print("true pos: ")
    print(true_positive)
    print("true neg: ")
    print(true_negative)
    print("false pos: ")
    print(false_positive)
    print("false neg: ")
    print(false_negative)
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    f1score = (2*precision*recall)/(precision+recall)
    print("precision: ")
    print(precision)
    print("recall: ")
    print(recall)
    print("accuracy: ")
    print(accuracy)
    print("f1score: ")
    print(f1score)
    
    #run and test random forest
    someForest = RandomForestClassifier()
    params = {'n_estimators': [100, 200, 300], 'min_samples_leaf': range(1,5), }
    clf = GridSearchCV(someForest, params)
    print("Random Forest--------------------")
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train)
    print(scores)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    true_positive = np.sum(np.logical_and(y_pred, y_test))
    true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    false_positive = 0
    false_negative = 0
    for i in range(0, y_test.size):
        if y_pred[i] and not y_test[i]:
            false_positive = false_positive + 1
        elif y_test[i] and not y_pred[i]:
            false_negative = false_negative + 1
    print("true pos: ")
    print(true_positive)
    print("true neg: ")
    print(true_negative)
    print("false pos: ")
    print(false_positive)
    print("false neg: ")
    print(false_negative)
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    f1score = (2*precision*recall)/(precision+recall)
    print("precision: ")
    print(precision)
    print("recall: ")
    print(recall)
    print("accuracy: ")
    print(accuracy)
    print("f1score: ")
    print(f1score)
    
    #logistic regression
    logistClass = LogisticRegression()
    params = {'C': [0.01, 0.1, 1.0, 10.0], 'fit_intercept': [False, True], 'max_iter': [500, 1000, 1500]}
    clf = GridSearchCV(logistClass, params)
    print("Logistic Regression--------------------")
    clf.fit(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train)
    print(scores)
    print(clf.best_params_)
    y_pred = clf.predict(X_test)
    true_positive = np.sum(np.logical_and(y_pred, y_test))
    true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    false_positive = 0
    false_negative = 0
    for i in range(0, y_test.size):
        if y_pred[i] and not y_test[i]:
            false_positive = false_positive + 1
        elif y_test[i] and not y_pred[i]:
            false_negative = false_negative + 1
    print("true pos: ")
    print(true_positive)
    print("true neg: ")
    print(true_negative)
    print("false pos: ")
    print(false_positive)
    print("false neg: ")
    print(false_negative)
    precision = true_positive/(true_positive+false_positive)
    recall = true_positive/(true_positive+false_negative)
    accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    f1score = (2*precision*recall)/(precision+recall)
    print("precision: ")
    print(precision)
    print("recall: ")
    print(recall)
    print("accuracy: ")
    print(accuracy)
    print("f1score: ")
    print(f1score)
    
    #print(sorted(clf.cv_results_.keys()))
    # print("Random forest score: ")
    # print(clf.score(X_test,y_test))
    
    #print(someForest.predict([[10,70,1,1,1,9,1,1,1,0]]))
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