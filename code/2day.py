#!/usr/bin/env python
# coding=utf-8


import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

dataset = pandas.read_csv('../datasets/studentscores.csv')

X = dataset.iloc[:,:1].values 
Y = dataset.iloc[:, 1].values

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_train_pred = regressor.predict(X_train)
Y_test_pred  = regressor.predict(X_test)


plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, Y_train_pred, color = 'blue')
plt.title('Training Set')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()

print(Y_test)
plt.scatter(X_test, Y_test, color = 'black')
plt.plot(X_test, Y_test_pred, color = 'green')
plt.title('Testing Set')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
