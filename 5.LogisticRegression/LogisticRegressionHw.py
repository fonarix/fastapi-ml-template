
# Grabbed from here: https://github.com/ayan-cs/logistic-regression-imdb/blob/main/Logistic_Regression_IMDb.ipynb

import numpy as np
import time
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('./IMDB Dataset.csv')
print(df.head(5))

# Prepare data:
num_samples = len(df['review'])
num_train = int(num_samples * 0.7)
num_test = int(num_samples * 0.3)
random_indices = np.random.permutation(num_samples)
     

X_train = df.loc[random_indices[:num_train], 'review'].values
y_train = df.loc[random_indices[:num_train], 'sentiment'].values
X_test = df.loc[random_indices[-num_test:], 'review'].values
y_test = df.loc[random_indices[-num_test:], 'sentiment'].values
     

print("X_train : ",X_train.shape," X_test : ",X_test.shape,"\ny_train : ",y_train.shape," y_test : ",y_test.shape)
     
# Model
# The model is built as a Pipeline of TF-IDF and Logistic Regression Classifier. At first, TF-IDF is computed and passed to the Classifier.

tfidf = TfidfVectorizer()
classifier = LogisticRegressionCV(max_iter = 4000)
clf = Pipeline([('tfidf',tfidf), ('clf',classifier)])
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
     
print(classification_report(y_test, y_pred))
     

'''
    negative       0.91      0.89      0.90      7496
    positive       0.90      0.91      0.90      7504

    accuracy                           0.90     15000
   macro avg       0.90      0.90      0.90     15000
weighted avg       0.90      0.90      0.90     15000

'''

result_accuracy =  accuracy_score(y_test, y_pred)

print("result_accuracy: ", result_accuracy)

'''
result_accuracy:  0.8982
'''

