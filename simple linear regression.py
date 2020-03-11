# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 19:29:52 2020

@author: Shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("Salary_Data.csv")   #importing dataset
x=df.iloc[:,:-1].values
y=df.iloc[:,1:2].values



#missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:,0])
x[:,0]=imputer.transform(x[:,0])

#encoding categorical value

"""from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
x[:,0]=labelencoder.fit_transform(x[:,0])

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[0])
x=onehotencoder.fit_transform(x).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)"""


#splitting dataset into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
x_train=standardscaler.fit_transform(x_train)
x_test=standardscaler.transform(x_test)
y_train=standardscaler.fit_transform(y_train)
y_test=standardscaler.transform(y_test)"""


#applying linear regression algorithm
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

#visualization of linear regression model

plt.scatter(x_train,y_train)
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()


plt.scatter(x_test,y_test)
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("salary vs experience")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()









