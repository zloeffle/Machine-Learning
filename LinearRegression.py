import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

'''
Goal: minimize the vertical dist between all the data points and our line
Mean Absolute Error is the easiest to understand, because it's the average error.
Mean Squared Error is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
Root Mean Squared Error is even more popular than MSE, because RMSE is interpretable in the "y" units.
'''
class LinearReg:
    def __init__(self, data):
        self.data = pd.read_csv(data)
    
    def details(self):
        data = self.data
        print(data.head())
        print(data.info())
        print(data.describe())
        
    def predict(self):
        df = self.data
        cols = list(df.columns)
        cols.remove('Price')
        cols.remove('Address')
        x = df[cols]
        y = df['Price']
        
        # SPLIT DATA
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4, random_state=101)
        
        # fit train/test data to our regression model
        lm = LinearRegression()
        lm.fit(x_train, y_train)

        # intercept
        print(lm.intercept_)
        
        # coeficients for each feature
        # means a 1 unit increase in col 1 corresponds to an increase of the item in col2
        print(lm.coef_)
        cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
        print(cdf)
    
        # PREDICTIONS
        pred = lm.predict(x_test)
        print('Predictions:\n',pred)
        plt.scatter(y_test, pred)
        #plt.show()
        
        # ERROR
        print('MAE:', metrics.mean_absolute_error(y_test, pred))
        print('MSE:', metrics.mean_squared_error(y_test, pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))


'''
Congratulations! You just got some contract work with an Ecommerce company based in New York City that sells clothing online but 
they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, 
then they can go home and order either on a mobile app or website for the clothes they want.

The company is trying to decide whether to focus their efforts on their mobile app experience or their website. They've hired you 
on contract to help them figure it out! Let's get started!
'''
class LinearReg_Example:
    def __init__(self, data):
        self.data = pd.read_csv(data)
    
    def details(self):
        data = self.data
        print(data.columns)
        print(data.shape)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum()) # get cols with null vals
        
    def predict(self):
        df = self.data
        y = df['Yearly Amount Spent']
        x = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']] # set to numerical features
        
        # SPLIT DATA
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=101)
        
        # fit train/test data to our regression model
        lm = LinearRegression()
        lm.fit(x_train, y_train)
        
        # coeficients for each feature
        # means a 1 unit increase in col 1 corresponds to an increase of the item in col2
        cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coefficient'])
        print(cdf)
        
        # PREDICTIONS
        pred = lm.predict(x_test)
        print('Predictions:\n',pred)
        plt.scatter(y_test, pred)
        plt.xlabel('Y Test True Values')
        plt.ylabel('Predicted Values')
        #plt.show()
        
        # ERROR
        print('MAE:', metrics.mean_absolute_error(y_test, pred))
        print('MSE:', metrics.mean_squared_error(y_test, pred))
        print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred)))

if __name__ == '__main__':
    lr = LinearReg('data/USA_Housing.csv')
    #lr.details()
    #lr.predict()
    
    # EXAMPLE
    lr = LinearReg_Example('data/Ecommerce Customers.csv')
    #lr.details()
    lr.predict()
    '''
    Do you think the company should focus more on their mobile app or on their website?

    This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app, 
    or develop the app more since that is what is working better. This sort of answer really depends on the other factors going on
    at the company, you would probably want to explore the relationship between Length of Membership and the App or the Website before coming to a conclusion!
    '''