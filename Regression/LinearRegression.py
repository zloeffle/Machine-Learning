from numpy.core.defchararray import mod
from numpy.lib.function_base import select
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
        self.alpha = alpha # learning rate
        self.n_iter = num_iter # number of iterations for the gradient descent
        self.n_features = np.size(x,1) # num of features
        self.n_samples = len(y) # num of rows 
        self.x = np.hstack((np.ones((self.n_samples,1)),(x-np.mean(x,0))/np.std(x,0))) # resca;e values for predictor vars
        self.y = y[:,np.newaxis] # increase dimension of target val array to imitate a col vector
        self.coef = None # coefficients
        self.params = np.zeros((self.n_features+1,1)) # initialize params as 0
        self.intercept = None

    '''
    Fit training data to the regression model to get the coefficients
    Also incorporates gradient descent optimization algorithm
    '''
    def fit(self):
        for i in range(self.n_iter):
            # compute the partial derivative of the cost function with respect to the parameters, update parameters
            self.params = self.params - (self.alpha/self.n_samples) * self.x.T @ (self.x @ self.params - self.y)
            
        self.intercept = self.params[0]
        self.coef = self.params[1:]

    '''
    Predict the target values using new data
    '''
    def predict(self,x):
        n_samples = np.size(x,0) # shape of new dataset
        y = np.hstack((np.ones((n_samples,1)),(x-np.mean(x,0))/np.std(x,0))).dot(self.params)
        return y
    
    '''
    Compute accuracy score using SSE
    '''
    def score(self,y_test,y_pred):
        y = y_test[:,np.newaxis]
        
        score = 1 - (((y-y_pred)**2).sum() / ((y-y.mean())**2).sum())
        return round(score,2)
    
if __name__ == '__main__':        
    df = pd.read_csv('data/USA_Housing.csv')
    print(df.head())
    print(df.info())
    print(df.describe())

    # clean data
    df.drop('Address',axis=1,inplace=True)
    
    # split into training and testing data
    x_train = df[df.columns[:-1]].iloc[:3500].to_numpy()
    y_train = df[df.columns[-1]].iloc[:3500].to_numpy()
    x_test = df[df.columns[:-1]].iloc[3500:].to_numpy()
    y_test = df[df.columns[-1]].iloc[3500:].to_numpy()
    
    model = LinearRegression(x_train,y_train)
    model.fit()
    pred = model.predict(x_test)
    score = model.score(y_test,pred)
    