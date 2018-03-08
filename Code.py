# Ch01：Python
from numpy import *
import numpy as np
import pandas as pd
import operator
from os import listdir
import matplotlib
import matplotlib.pylab as plt


a = random.rand(4, 4)  # 创建4*4数组
randMat = mat(a)  # 将数组转化为矩阵
invRandMat = randMat.I  # 求矩阵的逆
myEye = randMat * invRandMat  # 计算矩阵和矩阵的逆的乘积
myEye - eye(4)  # 计算误差 eye(4)生成4*4的单位矩阵


# Ch02：kNN
def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


group, labels = createDataSet()


# 2.1 kNN分类算法
def classify(inX, dataSet, labels, k):  # inX需分类的点，dataSet原始数据，数组格式，k类大小
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5  # 计算距离
    sortedDistIndicies = distances.argsort()  # 返回距离从小到大排序的索引值
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  # 统计距离最近的k个值中各类的数量
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  # 取距离最近的k个点，出现频率最高的类为该点的类
# test
# dataSet = group
# inX = [0, 0]
# k = 3
# inXclass = classify0(inX, group, labels, k)


# 2.2 使用k-近邻算法改进约会网站的配对效果
# 2.2.1解析数据
def file2matrix(DATAPATH, filename):
    frDataset = pd.read_table(DATAPATH + filename, header=None)  # 读取数据
    classLabelSeries = frDataset.pop(3)  # 取类标号
    classLabelVector = classLabelSeries.tolist()
    returnDataset = frDataset
    returnMat = np.array(returnDataset)  # 取数据并转化为数组
    return returnMat, classLabelVector


# 2.2.2 分析数据,做散点图
DATAPATH = '/Users/min/Documents/GitHub/machine-learning/machinelearninginaction/Ch02/'
filename = 'datingTestSet2.txt'
dataDatingMat, datingLabels = file2matrix(DATAPATH, filename)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataDatingMat[:, 0], dataDatingMat[:, 1], 20*np.array(datingLabels), np.array(datingLabels))
# scatter参数，1，2设定X.Y轴，第三个参数，设定点大小，第四个参数设定点的颜色
plt.show()


# 2.2.3 准备数据：归一化数值
# 将所有数值归一化到0-1之间的值，公式：newValue = (oldValue - min) / (max - min)
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    valDiff = maxVals - minVals
    normDataSet = dataSet - tile(minVals, (dataSet.shape[0], 1))
    normDataSet = normDataSet / tile(valDiff, (dataSet.shape[0], 1))
    return minVals, valDiff, normDataSet


# 2.2.4 测试算法
def datingClassTest(numTestVecs, normMat, m, k):
    errorCount = 0.0
    for i in range(numTestVecs):
        inX = normMat[i, :]
        classifierResult = classify(inX, normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], k)
        # print(
        # "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
        # print("the total error rate is: %f" % (errorCount / float(numTestVecs)))
    errorRate = errorCount / float(numTestVecs)
    # print(errorCount, errorRate)
    return errorRate


# 优化k取值，取误差最小的k
def ChoiceBestK():
    DATAPATH = '/Users/min/Documents/GitHub/machine-learning/machinelearninginaction/Ch02/'
    filename = 'datingTestSet2.txt'
    hoRatio = 0.10  # hold out 10%
    dataDatingMat, datingLabels = file2matrix(DATAPATH, filename)
    minVals, valDiff, normMat = autoNorm(dataDatingMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    maxK = m - numTestVecs
    errorRateDict = {}
    numTestVecs = int(m * hoRatio)
    for k in range(1, maxK):
        errorRate = datingClassTest(numTestVecs, normMat, m, k)
        errorRateDict[k] = errorRate
    # print(errorRateDict)
    errorRatesorted = sorted(errorRateDict.items(), key=operator.itemgetter(1), reverse=False)
    bestKvalue = errorRatesorted[0][0]
    bestErrorRate = errorRatesorted[0][1]
    return bestKvalue, bestErrorRate


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    inX = array([percentTats, ffMiles, iceCream])
    minVals, valDiff, normMat = autoNorm(dataDatingMat)
    bestKvalue, bestErrorRate = ChoiceBestK()
    classifierResult = classify((inX - minVals)/valDiff, normMat, datingLabels, bestKvalue)
    print('You will probably like this person : ', resultList[classifierResult - 1])


# 测试分类器
if __name__ == '__main__':
    bestKvalue, bestErrorRate = ChoiceBestK()
    classifyPerson()


# 2.3  手写识别系统》》示例
