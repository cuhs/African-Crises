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

europedata = DataManipulation.DataManipulation()
def main():
    readData = pd.read_csv("./Europe_crisis.csv")
    readData = readData.replace(' ', np.NaN)
    readData.replace('', np.NaN)
    readData = readData.dropna()
    readData = europedata.dropColumn(readData, readData.columns[0])
    readData = europedata.dropColumn(readData, 'CC3')
    readData = europedata.dropColumn(readData, 'Country')
    readData = europedata.dropColumn(readData, 'National Currency')
    readData = europedata.dropColumn(readData, 'exch_sources')
    minVal = europedata.findMin(readData, 'Year')
    readData['Year'] = europedata.subtractScalar(readData, 'Year', minVal)
    readData = readData.reset_index(drop=True)
    for i, row in readData.iterrows():
        gotStr = readData['exch_usd'].iloc[i].split("e",1)
        inflStr = readData['Inflation'].iloc[i]
        if(len(gotStr)>1):
            #print(gotStr[1])
            tens = pow(10,(float)(gotStr[1][1:len(gotStr[1])]))
            #print(tens)
            readData.at[i, 'exch_usd'] = float(gotStr[0])/tens
            #print(float(gotStr[0])/tens)
        else:
            readData.at[i, 'exch_usd'] = float(gotStr[0])
        #    print((float)(readData['exch_usd'].iloc[i]))
        readData.at[i, 'Inflation'] = float(inflStr)
        '''
        TODO: fix at not working, then get clipData and standardization working
        '''
    #readData = readData.astype('float')    
        
    #readData = europedata.clipData(readData, -500, 500)
    #readData['exch_usd'] = europedata.standardizeColumn(readData, 'exch_usd')
    #readData['Inflation'] = europedata.standardizeColumn(readData, 'Inflation')
    print(readData)
    readData.to_csv('./adjusted_european_crises_data.csv', index = True)


if __name__ == '__main__':
    main()