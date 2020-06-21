import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

'''
y = B0 + B1*X1 + ... + Bn*Xn
B = ((x_Transpose.x)^-1).x_Transpose.y
'''
class MultipleLinearRegression:
    def __init__(self):
        self.coef = [] # beta values
    
    def explore_data(self,data):
        print(data.head())
        print(data.info())
        print(data.describe())

    # split data into training and testing sets of specified size
    def train_test_split(self,x,y,size):
        split = int(len(x) * size) # point to split data

        # training and testing data for independent and dependent vars
        x_train,y_train = x.iloc[:split],y.iloc[:split]
        x_test,y_test = x.iloc[split:],y.iloc[split:]
        return x_train,y_train,x_test,y_test

    # adds column of 1's to data matrix
    def ones(self,x):
        ones = np.ones(shape=x.shape[0]).reshape(-1,1)
        return np.concatenate((ones,x),1)

    # fit training data to model
    def fit(self,x,y):
        # add col of ones to x matrix
        x = self.ones(x)
        # generate coeficients 
        coef = np.linalg.inv(x.transpose().dot(x)).dot(x.transpose()).dot(y)
        self.coef = coef

    # predict on new data
    def predict(self,x_test):
        preds = [] # predictions
        b0 = self.coef[0]
        betas = self.coef[1:]
        x_test = self.ones(x_test)

        for row in x_test:
            pred = b0
            for xi,bi in zip(row,betas):
                pred += xi*bi
            preds.append(pred)
        return preds
        
    
if __name__ == '__main__':        
    df = pd.read_csv('data/USA_Housing.csv')
    model = MultipleLinearRegression()
    #model.explore_data(df)

    # clean data
    df.drop('Address',axis=1,inplace=True)

    # split data
    cols = list(df.columns)
    x = df[cols[:-1]]
    y = df[cols[-1]]
    x_train,y_train,x_test,y_test = model.train_test_split(x,y,.6)
    
    # fit training data to model
    model.fit(x_train,y_train)

    # predict
    preds = model.predict(x_test)
    x_test['Actual'] = y_test
    x_test['Prediction'] = preds
    print(x_test)