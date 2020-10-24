import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class LogisticRegression:
    def __init__(self,x,y,alpha=0.03,num_iter=1500):
        self.alpha = alpha # learning rate
        self.n_iter = num_iter # number of iterations for the gradient descent 
        self.y = y[:,np.newaxis] # increase dimension of target val array to imitate a col vector
        self.n_samples = len(self.y) # num of rows
        self.x = np.hstack((np.ones((self.n_samples,1)),x))
        self.n_features = np.size(self.x,1) # num of features
        self.params = np.zeros((self.n_features,1)) # initialize params as 0


    '''
    Sigmoid function
    - Calculates probabilities of input data belonging to a certain class
    - outputs values 0..1
    '''
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))

    def gradient_descent(self):
        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * (self.x.T @ (self.sigmoid(self.x @ self.params) - self.y))

    def cost(self):
        pass

    '''
    Predict the target values using new data
    '''
    def predict(self,x_test):
        #n_samples = np.size(x_test,0)
        #x_test = np.hstack((np.ones((n_samples,1)),x_test))
        print(x_test)
        print(self.params)
        return self.sigmoid(x_test @ self.params)
    
    
if __name__ == '__main__':        
    df = pd.read_csv('data/advertising.csv')
    df.drop(['Ad Topic Line','City','Timestamp','Country'],axis=1,inplace=True)
    print(df.head())
    print(df.info())
    print(df.describe())

    
    x_train = df[df.columns[:-1]].iloc[:600].to_numpy()
    y_train = df[df.columns[-1]].iloc[:600].to_numpy()
    x_test = df[df.columns[:-1]].iloc[600:].to_numpy()
    y_test = df[df.columns[-1]].iloc[600:].to_numpy()
    
    print(x_train)
    print(y_train)

    model = LogisticRegression(x_train,y_train)
    model.gradient_descent()
    #pred = model.predict(x_test)
    #print(pred)
    