import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

'''
Representation of a node in the tree
'''
class Node:
    def __init__(self,feature,thresh,gini):
        self.feature = feature 
        self.threshold = thresh
        self.gini = gini # split cost
        self.left = None # left child
        self.right = None # right child
        self.is_terminal = False

class DecisionTree:
    def __init__(self,data,max_depth=5):
        self.max_depth = max_depth
        self.data = data

    def explore_data(self,data):
        print(data.head())
        print(data.info())
        print(data.describe())

    # prints a string representation of a tree
    def print(self,node,depth=0):
        if isinstance(node,Node):
            print('%s[%s < %.3f] Gini = %.4f' % (depth*' ',node.feature, node.threshold, node.gini))
            self.print(node.left, depth+1)
            self.print(node.right, depth+1)
        else:
            print('%s[%s]' % (depth*' ',node))

    # paritions data into left and right splits based on a feature and threshold
    def partition(self,data,feature,threshold):
        left,right = [],[]

        for i in list(data.index):
            if data.loc[i,feature] >= threshold:
                left.append(i)
            else:
                right.append(i)
        return left,right

    # gini = 1-sum[P_i * P_i]
    # calculates impurity of a data group
    def gini_impurity(self,data,groups):
        gini = 0.0
        instances = sum([len(group) for group in groups])
        target = list(data.columns)[-1]
        
        for group in groups:
            n = len(group)
            if n == 0:
                continue

            zero,one = 0,0
            for index in group:
                if data.loc[index,target] == 0:
                    zero += 1
                else:
                    one += 1
            gini += (1- (math.pow(zero/n,2) + math.pow(one/n,2))) * (n/instances)
        
        gini = round(gini,5)
        return gini

    # finds the optimal split in the dataset
    def best_split(self,data):
        best_feature,best_threshold,best_gini,best_groups = None,math.inf,math.inf,None
        features = list(data.columns[:-1])
        indicies = list(data.index)
        for col in features:
            for row in indicies:
                groups = self.partition(data,col,data.loc[row,col])
                gini = self.gini_impurity(data,groups)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = col
                    best_groups = groups
                    best_threshold = data.loc[row,col]

        node = Node(best_feature,best_threshold,best_gini)
        node.left = best_groups[0]
        node.right = best_groups[1]
        return node

    # makes a node terminal
    def to_terminal(self,group):
        cols = list(self.data.columns)
        zero,one = 0,0
        data = self.data

        for row in group:
            if data.loc[row,cols[-1]] == 0:
                zero += 1
            else:
                one += 1
        if zero > one:
            return 0
        return 1

    # generates left and right children for a node or makes it terminal
    def split_node(self,node,depth,min_size=1):
        left,right = node.left,node.right

        # check no split
        if left is None or right is None:
            node.left,node.right = self.to_terminal(left+right)
            return

        # check for max depth
        if depth >= self.max_depth:
            node.left = self.to_terminal(left)
            node.right = self.to_terminal(right)
            return

        # process left child
        if len(left) <= min_size:
            node.left = self.to_terminal(left)
        else:
            node.left = self.best_split(left)
            self.split_node(node.left, depth+1,min_size)

        # process right child
        if len(right) <= min_size:
            node.right = self.to_terminal(right)
        else:
            node.right = self.best_split(right)
            self.split_node(node.right, depth+1,min_size)

    # build the decision tree model
    def build_tree(self,data):
        root = self.best_split(data) # root node is optimal split
        self.split_node(root,10)
        return root

if __name__ == '__main__':
    df = pd.read_csv('data/Social_Network_Ads.csv')
    model = DecisionTree(df)
    #model.explore_data(df)

    # split data into training/testing sets
    cols = list(df.columns)
    x = df[cols[:-1]]
    y = df[cols[-1]]
    x_train,x_test = x.iloc[:300],x.iloc[300:]
    y_train,y_test = y.iloc[:300],y.iloc[300:]
    
    #left,right = model.partition(df,'Age')
    #model.gini_impurity(df,[left,right])
    #model.best_split(df)
    root = model.build_tree(df)
    model.print(root)