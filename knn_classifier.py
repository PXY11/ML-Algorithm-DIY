# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 19:39:51 2021

@author: Mr.P
"""

import numpy as np
from collections import Counter

def knn(input_vec,dataset,labels,k):
    '''
    INPUT:  
            input_vec: 待分类的样本
            dataset: 训练数据集
            labels: 训练数据集对应的标签
            k: 选择的邻近样本个数
    OUTPUT:
            predict: 样本最有可能所属的标签
    '''
    labels = np.array(labels)
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(input_vec,(dataset_size,1)) - dataset
    diff_mat = diff_mat**2
    distances = diff_mat.sum(axis=1) # axis=1 表示矩阵横向求和
    distances = distances**0.5
    dis_labels_mat = np.vstack((distances.T,labels.T))
    dis_labels_mat = dis_labels_mat.T
    sort_mat = dis_labels_mat[np.argsort(dis_labels_mat[:,0])] #将拼合好的[距离,标签]矩阵按第一列的距离排序 
    kmean_labels = sort_mat[:k,1] #取出前k个最小距离对应的标签
    collection_lbls = Counter(kmean_labels)
    predict = collection_lbls.most_common(1) #找出k个最小距离对应的标签中出现最多的标签，即为预测
    return predict[0][0]

group = np.array([[1,1],[1,1.1],[0,0],[0,0.1],[0.5,0.5]])
labels = np.array(['A','A','B','B','C'])
input_vec = np.array([0.5,0.9])

res = knn(input_vec,group,labels,3)
print('knn方法预测样本属于：%s'%res)
