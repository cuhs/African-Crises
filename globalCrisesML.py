from random import Random
import numpy as np
import pandas as pd
import DataManipulation
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

gd = DataManipulation.DataManipulation()
def main():
    readData = pd.read_excel("./global_crises_data.xlsx")
    readData = gd.dropColumn(readData, 'CC3')
    readData = gd.dropColumn(readData, 'Case')
    readData = gd.dropColumn(readData, 'exch_primary source code')
    readData = gd.dropColumn(readData, 'Defaults_External_Notes')
    readData = gd.dropColumn(readData, '<')
    readData = gd.dropColumn(readData, 'Banking_Crisis_Notes')
    readData = gd.dropColumn(readData, 'Country')
    readData = gd.dropColumn(readData, 'exch_sources')
    readData = gd.dropColumn(readData, 'exch_usd_alt1')
    readData = gd.dropColumn(readData, 'exch_usd_alt2')
    readData = gd.dropColumn(readData, 'exch_usd_alt3')
    readData = gd.dropColumn(readData, 'conversion_notes')
    readData = gd.dropColumn(readData, 'national currency')
    readData = gd.dropColumn(readData, 'Domestic_Debt_ Notes/Sources')
    readData = gd.dropColumn(readData, 'SOVEREIGN EXTERNAL DEBT 1: DEFAULT and RESTRUCTURINGS, 1800-2012--Does not include defaults on WWI debt to United States and United Kingdom and post-1975 defaults on Official External Creditors')
    readData = readData.replace(' ', np.NaN)
    readData = readData.dropna()
    currData = readData.loc[readData['Year'].isin([2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,2014])]
    oldData = readData.loc[~readData['Year'].isin([2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,2014])]
    readData = gd.dropColumn(readData, 'Year')
    currData = gd.dropColumn(currData, 'Year')
    oldData = gd.dropColumn(oldData, 'Year')
    readData.columns = readData.columns.str.strip()
    readData = readData.rename(columns = {'SOVEREIGN EXTERNAL DEBT 2: DEFAULT and RESTRUCTURINGS, 1800-2012--Does not include defaults on WWI debt to United States and United Kingdom but includes post-1975 defaults on Official External Creditors': 'Sovereign External Debt Default', 'GDP_Weighted_default': 'Weighted GDP', 'Inflation, Annual percentages of average consumer prices': 'Inflation'})
    #readData['Inflation'] = readData['Inflation'].str.strip()
    readData = readData.reset_index(drop=True)
    readData = readData.drop([138, 3991, 3992, 3993, 3994, 5407]) #blank indices that didnt delete before for some reason
    readData = readData.astype(float)
    readData = gd.clipData(readData, -500, 500)
    readData['exch_usd'] = gd.standardizeColumn(readData, 'exch_usd')
    readData['Inflation'] = gd.standardizeColumn(readData, 'Inflation')
    readData = gd.swapColumns(readData, 'Banking Crisis', 'Systemic Crisis')
    readData = gd.shuffleData(readData)
    
    X = readData.iloc[0:readData.shape[0], 1:readData.shape[1]].to_numpy()
    y = readData.iloc[0:readData.shape[0], 0].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
    
    #run and test decision tree
    # someTree = tree.DecisionTreeClassifier()
    # params = {'min_samples_split': range(2,15), 'min_samples_leaf': range(1,10), }
    # clf = GridSearchCV(someTree, params)
    # print("Decision Tree--------------------")
    # clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train)
    # print(scores)
    # print(clf.best_params_)
    # y_pred = clf.predict(X_test)
    # true_positive = np.sum(np.logical_and(y_pred, y_test))
    # true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    # false_positive = 0
    # false_negative = 0
    # for i in range(0, y_test.size):
    #     if y_pred[i] and not y_test[i]:
    #         false_positive = false_positive + 1
    #     elif y_test[i] and not y_pred[i]:
    #         false_negative = false_negative + 1
    # print("true pos: ")
    # print(true_positive)
    # print("true neg: ")
    # print(true_negative)
    # print("false pos: ")
    # print(false_positive)
    # print("false neg: ")
    # print(false_negative)
    # precision = true_positive/(true_positive+false_positive)
    # recall = true_positive/(true_positive+false_negative)
    # accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    # f1score = (2*precision*recall)/(precision+recall)
    # print("precision: ")
    # print(precision)
    # print("recall: ")
    # print(recall)
    # print("accuracy: ")
    # print(accuracy)
    # print("f1score: ")
    # print(f1score)
    
    # #run and test random forest
    # someForest = RandomForestClassifier()
    # params = {'n_estimators': [100, 200, 300], 'min_samples_leaf': range(1,5)}
    # clf = GridSearchCV(someForest, params)
    # print("Random Forest--------------------")
    # clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train)
    # print(scores)
    # print(clf.best_params_)
    # y_pred = clf.predict(X_test)
    # true_positive = np.sum(np.logical_and(y_pred, y_test))
    # true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    # false_positive = 0
    # false_negative = 0
    # for i in range(0, y_test.size):
    #     if y_pred[i] and not y_test[i]:
    #         false_positive = false_positive + 1
    #     elif y_test[i] and not y_pred[i]:
    #         false_negative = false_negative + 1
    # print("true pos: ")
    # print(true_positive)
    # print("true neg: ")
    # print(true_negative)
    # print("false pos: ")
    # print(false_positive)
    # print("false neg: ")
    # print(false_negative)
    # precision = true_positive/(true_positive+false_positive)
    # recall = true_positive/(true_positive+false_negative)
    # accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    # f1score = (2*precision*recall)/(precision+recall)
    # print("precision: ")
    # print(precision)
    # print("recall: ")
    # print(recall)
    # print("accuracy: ")
    # print(accuracy)
    # print("f1score: ")
    # print(f1score)
    
    #logistic regression
    # logistClass = LogisticRegression()
    # params = {'C': [0.01, 0.1, 1.0, 10.0], 'fit_intercept': [False, True], 'max_iter': [500, 1000, 1500]}
    # clf = GridSearchCV(logistClass, params)
    # print("Logistic Regression--------------------")
    # clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train)
    # print(scores)
    # print(clf.best_params_)
    # y_pred = clf.predict(X_test)
    # true_positive = np.sum(np.logical_and(y_pred, y_test))
    # true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    # false_positive = 0
    # false_negative = 0
    # for i in range(0, y_test.size):
    #     if y_pred[i] and not y_test[i]:
    #         false_positive = false_positive + 1
    #     elif y_test[i] and not y_pred[i]:
    #         false_negative = false_negative + 1
    # print("true pos: ")
    # print(true_positive)
    # print("true neg: ")
    # print(true_negative)
    # print("false pos: ")
    # print(false_positive)
    # print("false neg: ")
    # print(false_negative)
    # precision = true_positive/(true_positive+false_positive)
    # recall = true_positive/(true_positive+false_negative)
    # accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    # f1score = (2*precision*recall)/(precision+recall)
    # print("precision: ")
    # print(precision)
    # print("recall: ")
    # print(recall)
    # print("accuracy: ")
    # print(accuracy)
    # print("f1score: ")
    # print(f1score)
    
    
    # #ridge classifier
    # ridgeClass = RidgeClassifier()
    # params = {'alpha': [0.01, 0.1, 1, 10], 'normalize': [True, False], 'max_iter': [500]}
    # clf = GridSearchCV(ridgeClass, params)
    # print("Ridge Classifier--------------------")
    # clf.fit(X_train, y_train)
    # scores = cross_val_score(clf, X_train, y_train)
    # print(scores)
    # print(clf.best_params_)
    # y_pred = clf.predict(X_test)
    # true_positive = np.sum(np.logical_and(y_pred, y_test))
    # true_negative = np.sum(np.logical_not(np.logical_or(y_pred, y_test)))
    # false_positive = 0
    # false_negative = 0
    # for i in range(0, y_test.size):
    #     if y_pred[i] and not y_test[i]:
    #         false_positive = false_positive + 1
    #     elif y_test[i] and not y_pred[i]:
    #         false_negative = false_negative + 1
    # print("true pos: ")
    # print(true_positive)
    # print("true neg: ")
    # print(true_negative)
    # print("false pos: ")
    # print(false_positive)
    # print("false neg: ")
    # print(false_negative)
    # precision = true_positive/(true_positive+false_positive)
    # recall = true_positive/(true_positive+false_negative)
    # accuracy = (true_positive+true_negative)/(true_positive+false_positive+true_negative+false_negative)
    # f1score = (2*precision*recall)/(precision+recall)
    # print("precision: ")
    # print(precision)
    # print("recall: ")
    # print(recall)
    # print("accuracy: ")
    # print(accuracy)
    # print("f1score: ")
    # print(f1score)
    
    # #support vector machine
    SVMclass = SVC()
    params = {'C': [0.01, 0.1, 1], 'kernel': ['poly', 'rbf'], 'degree': range(2,3), 'tol': [0.001], 'max_iter': [1000]}
    clf = GridSearchCV(SVMclass, params)
    print("Support Vector Machine--------------------")
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
    
    readData.to_csv('./adjusted_global_crises_data.csv', index = True)
if __name__ == '__main__':
    main()