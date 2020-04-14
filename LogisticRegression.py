import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

'''
Example using titanic data
- Allows us to solve classification problems where we want to predict discrete categories
- Curve can only be between 0...1
- Sigmoid Function: sig(z) = 1/(1+e^-z), takes any val and output is 0...1
    - place linear reg soln into sigmoid function
    - linear model: y = b0 + b1*x
    - Logistic Model: p = 1 / (1 + e^-(b0 + b1*x))
'''
class LogisticReg:
    def __init__(self, data):
        self.data = pd.read_csv(data)

    def details(self):
        data = self.data
        print(data.columns + '\n')
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull())

        # heatmap to find missing data
        sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        
    # replaces missing vals in age column with the average age for the corresponding pclass
    def impute_age(self, cols):
        age = cols[0]
        pclass = cols[1]

        if pd.isnull(age):
            if pclass == 1:
                return 37
            elif pclass == 3:
                return 29
            else:
                return 24
        else:
            return age
    
    def clean(self):
        df = self.data
        # replace missing age vals
        df['Age'] = df[['Age', 'Pclass']].apply(self.impute_age, axis=1)
        
        # drop Cabin col since there are too many missing vals
        # also drop remaining missing data
        df.drop('Cabin', axis=1, inplace=True)
        df.dropna(inplace=True)
        
        # clean categorical columns
        sex = pd.get_dummies(df['Sex'],drop_first=True) # set drop_first as true since 1 col is perfect predictors of other col
        embarked = pd.get_dummies(df['Embarked'],drop_first=True)
        #print(sex, embarked)

        # combine sex & embarked cols
        # drop embarked, sex cols since we made new cols
        # drop ticket, name cols since they're unhelpful
        df = pd.concat([df, sex, embarked], axis=1)
        df.drop(['Sex','Embarked','Name','Ticket', 'PassengerId'], axis=1, inplace=True)
        
        print(df)
        self.data = df
    
    def predict(self):
        df = self.data
        x = df.drop('Survived',axis=1)
        y = df['Survived']
        
        # SPLIT DATA
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=101)
        lm = LogisticRegression() # logistic model
        lm.fit(x_train, y_train)
        
        # PREDICT
        pred = lm.predict(x_test)
        
        print(classification_report(y_test, pred))


'''
In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement on a company website.
We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
'''
class LogisticReg_Example:
    def __init__(self, data):
        self.data = pd.read_csv(data)

    def details(self):
        data = self.data
        print(data.columns + '\n')
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum()) # get cols with null vals
    
    def clean(self):
        df = self.data
        
        # drop useless cols
        df.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1, inplace=True)
        
        print(df)
        self.data = df
    
    def predict(self):
        df = self.data
        #self.clean() # no need to clean this dataset
        x = df[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
        y = df['Clicked on Ad']
        print(x,y)
        
        # SPLIT DATA
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=42)
        lm = LogisticRegression() # logistic model
        lm.fit(x_train, y_train)
        
        # PREDICT
        pred = lm.predict(x_test)
        
        print(pred)
        print(classification_report(y_test, pred))
        
if __name__ == '__main__':
    lr = LogisticReg('data/titanic_train.csv')
    '''
    df = lr.data
    # get average age of each class for impute_age
    c1 = df[df['Pclass'] == 1]
    c2 = df[df['Pclass'] == 2]
    c3 = df[df['Pclass'] == 3]
    print(c1['Age'].mean())
    print(c2['Age'].mean())
    print(c3['Age'].mean())
    lr.details()
    lr.clean()
    lr.predict()
    '''
    
    # EXAMPLE
    lr = LogisticReg_Example('data/advertising.csv')
    lr.details()
    lr.predict()