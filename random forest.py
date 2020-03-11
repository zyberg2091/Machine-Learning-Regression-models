# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 01:00:28 2020

@author: Shubh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))"""

#Random forest model
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(X,y)


y_pred=regressor.predict([[6.5]])
#visualization
x_grid=np.arange(min(X),max(X),0.01)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
