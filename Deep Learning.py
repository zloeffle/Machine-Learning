import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

'''
Example using bank_note_data.csv
Determines whether a bank note is authentic or not
'''
class DL:
    def __init__(self, data):
        self.data = pd.read_csv(data)
    
    def details(self):
        data = self.data
        print(data.columns)
        print(data.shape)
        print(data.head())
        print(data.info())
        print(data.describe())
        print(data.isnull().sum())
    
    def prep(self):
        df = self.data
        
        # instantiate scaler object & fit scaler to features
        scaler = StandardScaler()
        scaler.fit(df.drop('Class', axis=1))
        
        # transform the features to a scaled version
        scaled_feat = scaler.fit_transform(df.drop('Class', axis=1))
        
        # convert the scaled features to a df & check the head to ensure scaled worked properly
        df_feat = pd.DataFrame(scaled_feat,columns=df.columns[:-1])
        return df_feat
    
    def predict(self):
        df = self.data
        
        # scaled feature vals and labels
        x = self.prep()
        y = df['Class']
        
        # build training & testing sets of the data
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
        
        image_var = tf.feature_column.numeric_column("Image.Var")
        image_skew = tf.feature_column.numeric_column('Image.Skew')
        image_curt = tf.feature_column.numeric_column('Image.Curt')
        entropy =tf.feature_column.numeric_column('Entropy')
        feat_cols = [image_var,image_skew,image_curt,entropy]
        
        classifier = tf.estimator.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2,feature_columns=feat_cols)
        input_func = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=20,shuffle=True)
        classifier.train(input_fn=input_func,steps=500)
        
        pred_fn = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,batch_size=len(x_test),shuffle=False)
        note_pred = list(classifier.predict(input_fn=pred_fn))
        print(note_pred[0])
        
        final_preds  = []
        for pred in note_pred:
            final_preds.append(pred['class_ids'][0])
            
        print(confusion_matrix(y_test,final_preds))
        print(classification_report(y_test,final_preds))
        
        
if __name__ == '__main__':
    dl = DL('data/bank_note_data.csv')
    dl.details()
    dl.predict()