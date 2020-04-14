import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

'''
Unsupervised algorithm that groups similar clusters in the data together
- Ex: similar documents, customers based on features, market segmentation, identify similar physical groups

Goal
- divide data into distinct groups so observations in each group are similar

Algorithm
    1) choose num of clusters 'K'
    2) randomly assign each point to a cluster
    3) While clusters are not changing
        - for each cluster compute its centroid by taking the mean vector of points in the cluster
        - assign each point to the cluster for which the centroid is closest
'''
class KMeans_Clustering:
    def __init__(self, clusters=1):
        self.data = make_blobs(200,2,4,1.8,random_state=101) # 200 samples, 2 features, 4 centers
        self.clusters = clusters
        
    def plot(self):
        data = self.data
        plt.scatter(data[0][:,0], data[0][:,1], c=data[1], cmap='rainbow') # plot rows in 1st col vs rows in 2nd col
        plt.show()
    
    def build_clusters(self):
        data = self.data
        km = KMeans(4)
        km.fit(data[0])
        
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
        ax1.set_title('K Means')
        ax1.scatter(data[0][:,0],data[0][:,1],c=km.labels_,cmap='rainbow')
        ax2.set_title("Original")
        ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')
        plt.show()
        #km.cluster_centers_
        #km.labels_        
    
if __name__ == '__main__':
    kmc = KMeans_Clustering(4)
    kmc.build_clusters()