# -*-coding:utf-8-*-

"""
    Author: Labixiaohu
    Desc:利用KNN算法实现性别预测
"""
"""
KNN是一个分类算法，但是可以用于推荐，即为每个内容和物品寻找K个与其最相似的内容或物品进行推荐。
对性别预测也经常在电商中用，比如用户没有标注性别，防止把物品推荐错性别就要做性别的预测。
"""

import numpy as np

class KNN:
    def __init__(self, k):
        self.K = k

    # 准备数据
    def createData(self):
        featuers = np.array([[180,76],[158,43],[176,78],[161,49]])
        labels = ['男','女','男','女']
        return featuers, labels

    """
    由于身高体重并不在一个数据级上，所以做标准化Min-Max
    """
    def Normalization(self, data):
        maxs = np.max(data, axis=0)
        mins = np.min(data, axis=0)
        new_data = (data - mins) / (maxs - mins)
        return new_data, maxs, mins


    """
    计算K最近邻
    """
    def classify(self, one, data, labels):
        # 欧氏距离
        differenceData = data - one
        squareData = (differenceData ** 2).sum(axis=1)
        distance = squareData ** 0.5
        sortDistanceIndex = distance.argsort()
        # 统计K近邻的label
        labelCount = dict()
        for i in range(self.K):
            label = labels[sortDistanceIndex[i]]
            labelCount.setdefault(label, 0)
            labelCount[label] += 1
        # 计算结果
        sortLabelCount = sorted(labelCount.items(), key=lambda x:x[1], reverse=True)
        print(sortLabelCount)
        return sortLabelCount[0][0]


if __name__ == "__main__":
    knn = KNN(3)
    # 数据
    features, labels = knn.createData()
    # 标准化
    new_data, maxs, mins = knn.Normalization(features)
    # 待预测数据标准化
    one = np.array([176,76])
    new_one = (one - mins) / (maxs - mins)
    # 计算性别
    result = knn.classify(new_one,new_data,labels)
    print("数据{}的预测性别为：{}".format(one,result))