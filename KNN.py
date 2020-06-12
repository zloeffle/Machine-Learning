import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split

'''
Training - Store all data
Prediction for new points
    1) calc dist from x to all points in the data
    2) sort points by increasing dist from x
    3) predict the majority label of the k closest points
Pros
- Simple, training is trivial, works with any num of classes, easy to add new data, only params are K and distance metric
Cons
- High prediction cost, worse for large data set
- not good with high dimension data
- categorical features dont work well
'''

class KNN:
    def __init__(self,data):
        self.data = pd.read_csv(data, index_col=0)

    def details(self):
        data = self.data
        print(data.columns)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum()) # get cols with null vals

    # Standardize Variables
    def standardize(self):
        df = self.data
        
        # create instance of scaler obj
        scaler = StandardScaler()
        
        # fit scaler to our data
        scaler.fit(df.drop('TARGET CLASS',axis=1))
        
        # transform data
        scaled_feat = scaler.transform(df.drop('TARGET CLASS',axis=1))
        print(scaled_feat)
        return scaled_feat
        
    def classify(self, neighbors=1):
        df = self.data
        scaled_feat = self.standardize()
        x_train,x_test,y_train,y_test = train_test_split(scaled_feat,df['TARGET CLASS'],test_size=0.30)
    
        knn = KNeighborsClassifier(neighbors)
        knn.fit(x_train, y_train)
        
        pred = knn.predict(x_test)

        print(confusion_matrix(y_test,pred))
        print(classification_report(y_test,pred))
        return x_train,y_train,x_test,y_test
        
    # helps choosing a k that results in lowest error rate
    def choose_k(self):
        error = []
        data = self.classify()
        
        for i in range(1,40):
            knn = KNeighborsClassifier(i)
            knn.fit(data[0], data[1])
            pred_i = knn.predict(data[2])
            error.append(np.mean(pred_i != data[3]))
        
        plt.figure(figsize=(10,6))
        plt.plot(range(1,40),error,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        plt.show()
        
if __name__ == '__main__':
    knn = KNN('data/Classified Data.csv')
    #knn.details()
    knn.standardize()
    #knn.classify()
    #knn.choose_k()
    