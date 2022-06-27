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
    

