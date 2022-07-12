# -*- coding: utf-8 -*-
"""churn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xlYHEVkg7JfVpTG_9Lga006FgSdjqyco
"""

import pickle
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectPercentile, chi2
url = 'churn.csv'

df = pd.read_csv(url, delimiter=',')

cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

convert = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'Churn']

df[convert] = df[convert].astype('category')

X = df.iloc[:,1:-1]
y = df['Churn']

trans = make_column_transformer(
    (OneHotEncoder(),['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod']),
    (SimpleImputer(strategy='median'), ['TotalCharges'] ),
    remainder='passthrough'
)

selector = SelectPercentile(chi2, percentile=65)


pipe = make_pipeline(trans, selector, LogisticRegressionCV(max_iter=1000, solver='lbfgs'))
print(cross_val_score(pipe, X, y, cv=5,scoring='accuracy').mean())

pipe.fit(X,y)

pickle.dump(pipe, open('churn_pipe.pkl', 'wb'))




