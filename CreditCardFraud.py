# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:21:19 2020

@author: Maneshwar
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

df= pd.read_csv('creditcard.csv')
df.shape
df.head()
df.info()

df['Amount'].mean()
df['Amount'].max()

plt.scatter(x=df['Amount'], y=df['Time'])
plt.hist(df['Amount'])

df.loc[:, ['Time', 'Amount']].describe()
plt.figure(figsize=(10, 8))

sns.distplot(df['Time'])
sns.distplot(df.Amount)

counts= df['Class'].value_counts()
normal= counts[0]
fradulent= counts[1]

perc_normal= normal/(normal+fradulent)*100

df_counts=counts.to_frame()
df_counts['Count']=[0, 1]

plt.figure(figsize=(8, 6))
sns.barplot(x=df_counts.iloc[:, 1], y=df_counts.iloc[:, 0])

corr=df.corr()
plt.figure(figsize=(12, 10))
heat= sns.heatmap(data=corr)
plt.title('Correlation Heatmap')

skew_= df.skew()

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
scaled_time= scaler.fit_transform(df[['Time']])

scaled_time_list = [item for sublist in scaled_time.tolist() for item in sublist]
scaled_time= pd.Series(scaled_time_list)

scaler2= StandardScaler()
scaled_amount= scaler2.fit_transform(df[['Amount']])
scaled_amount_list = [item for sublist in scaled_amount.tolist() for item in sublist]
scaled_amount=pd.Series(scaled_amount_list)

df= pd.concat([df, scaled_time.rename('scaled_time'), scaled_amount.rename('scaled_amount')], axis= 1)
df.head()

df.drop(['Amount', 'Time'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split
train, test= train_test_split(df, test_size= 0.1)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

no_of_frauds= train.Class.value_counts()[1]

non_fraud = train[train['Class'] == 0]
fraud = train[train['Class'] == 1]
selected= non_fraud.sample(no_of_frauds)
selected.head()
selected.reset_index(inplace=True, drop=True)

subsample= pd.concat([selected, fraud])
subsample= subsample.sample(frac=1).reset_index(drop=True)

new_counts= subsample.Class.value_counts()
plt.figure(figsize=(10, 8))
sns.barplot(x=new_counts.Index, y=new_counts.Class)

corr= subsample.corr()
corr= corr[['Class']]
corr[corr.Class<-0.5]
corr[corr.Class>0.5]

Q1 = subsample.quantile(0.25)
Q3 = subsample.quantile(0.75)
IQR = Q3 - Q1

df2 = subsample[~((subsample < (Q1 - 2.5 * IQR)) |(subsample > (Q3 + 2.5 * IQR))).any(axis=1)]

from sklearn.manifold import TSNE

X = df2.drop('Class', axis=1)
y = df2['Class']

plt.scatter(x=X['scaled_time'], y=X['scaled_amount'])

X_reduced_tsne= TSNE(n_components= 2, random_state= 42).fit_transform(X.values)

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn= warn

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2, random_state=42)

X_train= X_train.values
X_validation= X_test.values
y_train= y_train.values
y_test= y_test.values

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

models=[]
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))
models.append(('XGB', XGBClassifier()))
models.append(('RF', RandomForestClassifier()))

results=[]
names=[]

for name, model in models:
    kfold= KFold(n_splits= 10, random_state=42)
    cv_results= cross_val_score(model, X_train, y_train, cv=kfold, scoring='roc_auc')
    names.append(name)
    results.append(cv_results)
    print('%s %f %f' % (name, cv_results.mean(), cv_results.std()))


model= RandomForestClassifier(n_estimators=10)
model.fit(X_train, y_train)











































