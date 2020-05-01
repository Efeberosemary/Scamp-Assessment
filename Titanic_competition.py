# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 00:52:27 2020

@author: EFEBE MARK
"""

import pandas as pd
# importing the training data
train_data = pd.read_csv(r'C:\Users\EFEBE MARK\Documents\titanic competition\train.csv')
# selecting the necessary data for model fitting
column_target =['Pclass','Sex','Age','Fare']
X= train_data[column_target]
#replacing the null in the age category with the median of the ages
X['Age']=X['Age'].fillna(X['Age'].median())

#transforming sex data into binary format
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Xfit =X[X.columns[:5]].apply(le.fit_transform)

# assigning the target to the Y variable
Y =train_data['Survived']

#training the model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 180, min_samples_leaf=3, max_features = 0.5, n_jobs = -1)
model.fit(Xfit,Y)

# importing the testing set
test_data  = pd.read_csv(r'C:\Users\EFEBE MARK\Documents\titanic competition\test.csv')
# preprocessing the testing data
column_target =['Pclass','Sex','Age','Fare']
Xtest= test_data[column_target]
Xtest['Age']= Xtest['Age'].fillna(Xtest['Age'].mean())
Xtestfit= Xtest[Xtest.columns[:5]].apply(le.fit_transform)

# carrying out prediction
Ypred = model.predict(Xtestfit)

# commiting the oredicted data to a csv file
survivors =pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':Ypred})
kaggle = survivors.to_csv('Kaggle_submittionnew.csv',
                          columns=['PassengerId','Survived'], index = False)


