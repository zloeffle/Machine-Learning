from numpy.core.defchararray import mod
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

'''
Method for modeling the past relationship between independent input variables and dependent output variables to help make predictions about the output variables in the future. 
- modeling equation: y = B0 + B1*X1 + ... + Bn*Xn
- coeficients calculated by: B = ((x_Transpose.x)^-1).x_Transpose.y
    - where x = input variables and y = target variable

Uses
- Optimizing product level price points
- Estimating price elasticities
- Analyzing metrics driving product sales (pricing,volume,etc)
'''
class LinearRegression:
    def __init__(self,x,y,alpha=0.03,num_iter=1500):
        self.alpha = alpha
        self.n_iter = num_iter
        self.n_features = np.size(x,1)
        self.n_samples = len(y)
        self.x = np.hstack((np.ones((self.n_samples,1)),(x-np.mean(x,0))/np.std(x,0)))
        self.y = y[:,np.newaxis]
        self.coef = None
        self.params = np.zeros((self.n_features+1,1))
        self.intercept = None

    # fit training data to model
    def fit(self):
        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * self.x.T @ (self.x @ self.params - self.y)
            
        self.intercept = self.params[0]
        self.coef = self.params[1:]

    # predict on new data
    def predict(self,x):
        n_samples = np.size(x,0)
        y = np.hstack((np.ones((n_samples,1)),(x-np.mean(x,0))/np.std(x,0))) @ self.params
        return y
        
    
if __name__ == '__main__':        
    df = pd.read_csv('data/USA_Housing.csv')
    print(df.head())
    print(df.info())
    print(df.describe())

    # clean data
    df.drop('Address',axis=1,inplace=True)
    
    # split into independent and dependent vars
    x_train = df[df.columns[:-1]].iloc[:3500].to_numpy()
    y_train = df[df.columns[-1]].iloc[:3500].to_numpy()
    x_test = df[df.columns[:-1]].iloc[3500:].to_numpy()
    
    model = LinearRegression(x_train,y_train)
    print(model.params)
    model.fit()
    pred = model.predict(x_test)
    
    df = df.iloc[3500:]
    df['Pred Price'] = pred
    print(df)