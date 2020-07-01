import pandas as pd
import numpy as np
import math
import random

'''
Classifies a set of data based on the k-closest labels from a training dataset
'''
class KNN_Classifier:
    def __init__(self,k):
        self.k = k # number of neighbors used for classification

    '''
    Calculates eulicidean distance between two data points
    
    Params: v1,v2 = vectors representing two data points
    Returns: float
    '''
    def eulicidean_distance(self,v1,v2):
        dist = 0
        v1 = np.array(v1)
        v2 = np.array(v2)

        for i,j in zip(v1,v2):
            dist += math.pow(i-j,2)
        return float(math.sqrt(dist))

    '''
    Params: train = training data points, test = test data point
    Returns: list of k closest data points to the testing data point 
    '''
    def get_neighbors(self,train,test):
        dist = []       
        cols = list(train.columns)

        for row in train.index:
            temp = self.eulicidean_distance(test,train.loc[row,cols[:-1]])
            dist.append((row,temp,train.loc[row,cols[-1]]))    
        dist.sort(key=lambda x: x[1])
        dist = dist[:self.k]
        return dist
    '''
    Params: list of data points
    Returns: most frequent label from input list
    '''
    def classify(self,neighbors):
        labels = {}
        for n in neighbors:
            if n[2] not in labels:
                labels[n[2]] = 1
            else:
                labels[n[2]] += 1
        most_freq = max(labels,key=labels.get)
        return most_freq

if __name__ == '__main__':
    model = KNN_Classifier(5)
    data = pd.read_csv('data/Classified Data.csv')
    data.drop(data.columns[0],axis=1,inplace=True)
    
    cols = list(data.columns)
    x_test = data[cols[:-1]].iloc[:5]
    y_test = data[cols[-1]].iloc[:5]
    train = data[cols].iloc[5:]
    
    for i in x_test.index:
        n = model.get_neighbors(train,x_test.loc[i,cols[:-1]])
        classification = model.classify(n)
        