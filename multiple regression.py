# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 22:14:05 2020

@author: Shubh
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("50_Startups.csv")   #importing dataset
df

x=df.iloc[:,:4].values
y=df.iloc[:,-1:].values



#missing values
from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values="NaN",strategy="mean",axis=0)
imputer=imputer.fit(x[:,:3])
x[:,:3]=imputer.transform(x[:,:3])

#encoding categorical value

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
x[:,3]=labelencoder.fit_transform(x[:,3])

from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_features=[3])
x=onehotencoder.fit_transform(x).toarray()


#Dummy variable trap
x=x[:,1:]


#splitting dataset into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
"""from sklearn.preprocessing import StandardScaler
standardscaler=StandardScaler()
x_train=standardscaler.fit_transform(x_train)
x_test=standardscaler.transform(x_test)

y_train=standardscaler.fit_transform(y_train)
y_test=standardscaler.transform(y_test)"""


#multiple linear algo
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)


#backward elimination method
from statsmodels.regression.linear_model import OLS
x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,:6]
regressor_OLS=OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()

x_opt=x[:,[0,3]]
regressor_OLS=OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()


#splitting dataset into train and test dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_opt,y,test_size=0.2,random_state=0)

#multiple linear algo
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor=regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

#visualizationsSSSS









