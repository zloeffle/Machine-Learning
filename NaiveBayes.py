import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import math

'''
Ex: mark email as spam/not, classify news article, classify text as positive/negative emotion, facial recognition
Bayes Theorm: P(A|B) = (P(B|A) * P(A)) / P(B)

Example of a Naive Bayes classifier using the iris dataset
'''

class NaiveBayes:
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = None
        
    # prepare the data to be converted into a dataframe
    def prep(self):
        dataset = self.dataset
        columns = ['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width']
        data = pd.DataFrame(dataset['data'], columns=columns)
        data['Species'] = dataset['target']
        data['Species'] = data['Species'].apply(lambda x: dataset['target_names'][x])
        self.data = data
        
    def details(self):
        print(self.data.head)
        print(self.data.shape)
        print(self.data.describe)
        print(self.data.info)
        print(self.data.isnull().sum())

    # 1) Seperate training data by class, returns dictionary of seperated data
    def seperate(self):
        data = self.data
        seperated = {}
        for i,row in data.iterrows():
            vector = list(row)
            class_val = vector[-1]
            if class_val not in seperated:
                seperated[class_val] = []
            seperated[class_val].append(vector)
        return seperated
        
    # 2) Summarize dataset
    def mean(self, data):
        return sum(data)/float(len(data))
    
    def stdev(self, data):
        avg = self.mean(data)
        variance = sum([(x-avg)**2 for x in data])/float(len(data)-1)
        return math.sqrt(variance)
        
    def summarize(self,dataset=None):
        data = self.data[['Petal Length', 'Petal Width', 'Sepal Length', 'Sepal Width']]
        summaries = []
        
        for key,val in data.iteritems():
            summaries.append((self.mean(list(val)), self.stdev(list(val)), len(val)))
        del(summaries[-1])
        print(summaries)
        return summaries
    
    # 3) Summarize dataset organized by class
    def summarize_class(self):
        seperated = self.seperate()
        summaries = {}
        for val, rows in seperated.items():
            for item in rows:
                item.pop()
            summaries[val] = [(self.mean(col), self.stdev(col), len(col)) for col in zip(*rows)]
        return summaries
        
    # 4) Gaussian probability
    def calc_prob(self, data, mean, stdev):
        exponent = math.exp(-((data-mean)**2 / (2 * stdev**2 )))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent
        
    # 5) Class Probabilities
    def class_probs(self, row):
        summaries = self.summarize_class()
        rows = sum([summaries[label][0][2] for label in summaries])
        probs = {}
        for val, class_sum in summaries.items():
            probs[val] = summaries[val][0][2]/float(rows)
            for i in range(len(class_sum)):
                mean,stdev,_ = class_sum[i]
                probs[val] *= self.calc_prob(row[i], mean, stdev)
        return probs
    
    def predict(self, row):
        probs = self.class_probs(row)
        best_label, best_prob = None, -1
        for val, prob in probs.items():
            if best_label is None or prob > best_prob:
                best_prob = prob
                best_label = val
        return best_label
        
if __name__ == '__main__':
    nb = NaiveBayes(load_iris())
    nb.prep()
    seperated = nb.seperate()
    for label in seperated:
        print(label)
        for row in seperated[label]:
            print(row)
    summary = nb.summarize_class()
    for label in summary:
	    print(label)
	    for row in summary[label]:
		    print(row)
    
    print(nb.calc_prob(1.0, 1.0, 1.0))
    print(nb.calc_prob(2.0, 1.0, 1.0))
    print(nb.calc_prob(0.0, 1.0, 1.0))

    print(nb.predict([6.5, 3.0, 5.2, 2.0]))
    