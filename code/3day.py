#!/usr/bin/env python
# coding=utf-8

import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split

dataset = pandas.read_csv('../datasets/50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

labelencoder = LabelEncoder()
X[: , 3]  = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

X = X[: , 1:]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

plt.scatter(numpy.arange(10), Y_test, color = 'red')
plt.scatter(numpy.arange(10), Y_pred, color = 'blue')

plt.legend(loc=2)
plt.show()
