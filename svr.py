# -*- coding: utf-8 -*-

Created on Mon Feb 17 23:01:41 2020

@author: Shubh


# Importing the dataset
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
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
sc_y = StandardScaler()
y = sc_y.fit_transform(y.reshape(-1,1))

#SVR model
from sklearn.svm import SVR
regressor=SVR(kernel='rbf')
regressor.fit(X,y)


y_pred=sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#visualization
x_grid=np.arange(min(X),max(X),0.1)
x_grid=x_grid.reshape(len(x_grid),1)
plt.scatter(X,y,color='red')
plt.plot(x_grid,regressor.predict(x_grid),color='blue')
file:///C:/Users/Shubh/Downloads/competitive-data-science-predict-future-sales.zip