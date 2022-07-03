import numpy as np
import pandas as pd
import DataManipulation
from sklearn.linear_model import Perceptron

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
    X = readData.iloc[0:readData.shape[0], 1:readData.shape[1]]
    #slice out y vector
    y = readData.iloc[0:readData.shape[0], 0]

    #africadata.writeToCsv(readData, './adjusted_african_crises_data.csv')

    

def runPerceptron(): 
    acPerceptron = Perceptron(
        max_iter = 200,
        tol = .01
    )
    acPerceptron.fit()
if __name__ == '__main__':
    main()