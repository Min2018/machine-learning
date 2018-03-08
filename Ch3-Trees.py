from numpy import *
import pandas as pd
from math import log


# 计算信息增益/香农熵
def calcSchannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCount = {}
    for line in dataSet:
        label = line[-1]
        if label not in labelCount.keys():
            labelCount[label] = 0
        labelCount[label] += 1  # 计算各类的数量
    schannonEnt = 0.0
    for key in labelCount:
        prob = float(labelCount[key]/numEntries)
        schannonEnt -= prob * log2(prob)
    return schannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    #change to discrete values
    return dataSet, labels


dataSet, labels = createDataSet()
calcSchannonEnt(dataSet)

