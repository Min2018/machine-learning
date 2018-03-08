from numpy import *
import pandas as pd
from math import log


# 计算信息增益/香农熵
def calcSchannonEnt(dataSet):
    numEntries = len(dataSet)

