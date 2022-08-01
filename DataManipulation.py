import numpy as np
import pandas as pd


class DataManipulation:
    def dropColumn(self, table, label):
        return table.drop(label, 1)
    def findMin(self, table, label):
        return table[label].min()
    def subtractScalar(self, table, label, value):
        return table[label] - value
    def replaceStrWithInt(self, table, label, str, int):
        return table[label].replace(str, int)
    def findMean(self, table, label):
        return table[label].mean()
    def findStd(self, table, label):
        return table[label].std()
    def standardizeColumn(self, table, label):
        return (table[label] - table[label].mean())/table[label].std()
    def writeToCsv(self, table, path):
        table.to_csv(path, index = False)
    def clipData(self, table, min, max):
        return table.clip(min, max)
    def swapColumns(self, table, col1, col2):
        cols = list(table.columns)
        a, b = cols.index(col1), cols.index(col2)
        cols[b], cols[a] = cols[a], cols[b]
        table = table[cols]
        return table
    def shuffleData(self, table):
        return table.sample(frac = 1)
    def splitString(self, str, delim):
        return str.split(delim)
    

