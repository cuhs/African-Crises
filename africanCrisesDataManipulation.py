import numpy as np
import pandas as pd
import DataManipulation

africadata = DataManipulation.DataManipulation()
def main():
    readData = pd.read_csv("./african_crises_raw_data.csv")
    readData = africadata.dropColumn(readData, 'case')
    readData = africadata.dropColumn(readData, 'cc3')
    readData = africadata.dropColumn(readData, 'country')
    minVal = africadata.findMin(readData, 'year')
    readData['year'] = africadata.subtractScalar(readData, 'year', minVal)
    readData['banking_crisis'] = africadata.replaceStrWithInt(readData, 'banking_crisis', "crisis", 1)
    readData['banking_crisis'] = africadata.replaceStrWithInt(readData, 'banking_crisis', "no_crisis", 0)
    readData['exch_usd'] = africadata.standardizeColumn(readData, 'exch_usd')
    readData['inflation_annual_cpi'] = africadata.standardizeColumn(readData, 'inflation_annual_cpi')
    print(readData['inflation_annual_cpi'])
    
if __name__ == '__main__':
    main()