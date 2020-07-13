import pandas as pd
import numpy as np
import math
import random

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

'''
Classifies a set of data based on the k-closest labels from a training dataset
'''
class KNN_Classifier:
    def __init__(self,k):
        self.k = k # number of neighbors used for classification

    def scale_data(self,data):
        scaler = StandardScaler()
        scaler.fit(data.drop('TARGET CLASS',axis=1))
        scaled_feat = scaler.transform(data.drop('TARGET CLASS',axis=1))
        res = pd.DataFrame(scaled_feat,columns=data.columns[:-1])
        return res        
        
    def split_data(self,data):
        scaled_data = self.scale_data(data)
        x_train,x_test,y_train,y_test = train_test_split(scaled_data,data['TARGET CLASS'],test_size=0.30,random_state=101)
        return x_train,x_test,y_train,y_test
    
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
    cols = list(data.columns)
    data.drop(data.columns[0],axis=1,inplace=True)
    
    # split data and scale features
    x_train,x_test,y_train,y_test = model.split_data(data)
    
    # get neighbors
    for row in x_test.index:
        neighbors = model.get_neighbors(x_train,x_test.loc[row,:])
        classification = model.classify(neighbors)
        #x_test.loc[row,'PREDICTED CLASS'] = classification
    #x_test['TARGET CLASS'] = y_test
    #print(x_test)