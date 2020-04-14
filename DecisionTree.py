import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn.tree import export_graphviz
import pydot

'''
Example using kyphosis dataset

Nodes
- split for the value of an attribute
Leaves
- terminal nodes that predict the outcome
Entropy
- H(S) = -sum[p_i(S) * log_2(p_i(S))]
Info Gain
- IG(S,A) = H(S) - sum v->Vals(A)[(S_v/S) * H(S_v)]
Random Forests
- to improve performance, we can use many trees with a random sample of features chosen as the split
- a new rand sample of features is chosen for every tree at every split
- m = sqrt(p)
'''
class DecisionTree:
    def __init__(self, data):
        self.data = pd.read_csv(data)
        
    def details(self):
        data = self.data
        print(data.columns)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum()) 

    def predict(self):
        df = self.data
        
        x = df.drop('Kyphosis', axis=1)
        y = df['Kyphosis']
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30)
        
        dt = DecisionTreeClassifier()
        dt.fit(x_train, y_train)
        dt_pred = dt.predict(x_test)
        
        rf = RandomForestClassifier(100)
        rf.fit(x_train, y_train)
        rf_pred = rf.predict(x_test)
        
        print(dt_pred, rf_pred)
        
        print(confusion_matrix(y_test,dt_pred))
        print(classification_report(y_test,dt_pred))
        print(confusion_matrix(y_test,rf_pred))
        print(classification_report(y_test,rf_pred))
        return dt,rf
        
if __name__ == '__main__':
    dt = DecisionTree('data/kyphosis.csv')
    dt.details()
    dt.predict()