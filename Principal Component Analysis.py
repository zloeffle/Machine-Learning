import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
PCA is an unsupervised algorithm that attempts to find out what features explain the most variance in your data
'''

class PCA_Ex:
    def __init__(self):
        self.data = load_breast_cancer()
        self.prep()
        
    def prep(self):
        data = self.data
        self.data = pd.DataFrame(data['data'], columns=data['feature_names'])
    
    def details(self):
        print(self.data.head())
        print(self.data.shape)
        print(self.data.describe())
        print(self.data.info())
        print(self.data.isnull().sum())
        
    def predict(self):
        df = self.data
        
        # scale data so each feature has a single unit variance
        scaler = StandardScaler()
        scaler.fit(df)
        scaled_data = scaler.transform(df)
        
        # find the principal components using fit, the apply the rotation in dimensionality reduction
        pca = PCA(n_components=2) # keep two components
        pca.fit(scaled_data)
        x_pca = pca.transform(scaled_data) # transform data to its first principal components
        
        
        
        # shows that using the two components we kept, we can seperate the two classes
        data = load_breast_cancer()
        print(pca.components_)
        df_comp = pd.DataFrame(pca.components_, columns=data['feature_names'])
        print(df_comp)
        plt.scatter(x_pca[:,0],x_pca[:,1],c=data['target'],cmap='plasma')
        plt.xlabel('First principal component')
        plt.ylabel('Second Principal Component')
        plt.show()
        
if __name__ == '__main__':
    pca = PCA_Ex()
    pca.details()
    pca.predict()