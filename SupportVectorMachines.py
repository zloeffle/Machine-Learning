import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

'''
SVM
- supervised learning models with learning algorithms that analyze data and recognize patterns used for classification and regression analysis
- builds a model that assigns new examples into one category or the other, thus making it a non-probabilistic binary linear classifier
'''

# uses breast_cancer dataset
# predicts whether tumors are benign or malignant
class SVM:
    def __init__(self):
        self.data = load_breast_cancer()
        self.df_feat = None
        self.df_target = None
        self.prep()
        
    def prep(self):
        data = self.data
        self.df_feat = pd.DataFrame(data['data'], columns=data['feature_names'])
        self.df_target = pd.DataFrame(data['target'], columns=['Cancer'])
    
    def details(self):
        print(self.df_feat.head())
        print(self.df_feat.shape)
        print(self.df_feat.describe())
        print(self.df_feat.info())
        print(self.df_feat.isnull().sum())
    
    def predict(self):
        x_train,x_test,y_train,y_test = train_test_split(self.df_feat,np.ravel(self.df_target),test_size=0.3, random_state=101)
        model = SVC()
        model.fit(x_train, y_train)
        
        pred = model.predict(x_test)
        print(confusion_matrix(y_test,pred))
        print(classification_report(y_test,pred))
        print(pred)
        return pred

'''
Uses iris dataset, predicts species of flower
'''
class SVM_Iris:
    def __init__(self):
        self.data = load_iris()
        self.prep()
        
    def prep(self):
        dataset = self.data
        columns = ['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width']
        data = pd.DataFrame(dataset['data'], columns=columns)
        data['Species'] = dataset['target']
        data['Species'] = data['Species'].apply(lambda x: dataset['target_names'][x])
        self.data = data
    
    def details(self):
        data = self.data
        print(data.columns)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum())
    
    def predict(self):
        df = self.data

        x = df.drop('Species', axis=1)
        y = df['Species']
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

        model = SVC()
        model.fit(x_train, y_train)        
        pred = model.predict(x_test)
       
        print(confusion_matrix(y_test,pred))
        print(classification_report(y_test,pred))
        print(pred)
        return pred

if __name__ == '__main__':
    svm = SVM()
    #svm.details()
    #svm.predict()
    
    svm = SVM_Iris()
    svm.details()
    svm.predict()