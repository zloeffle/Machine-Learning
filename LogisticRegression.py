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
class LogisticRegression:
    def __init__(self):
        pass
        
if __name__ == '__main__':
    