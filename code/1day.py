#!/usr/bin/env python
# coding=utf-8

import numpy
import pandas


dataset = pandas.read_csv('../datasets/Data.csv')

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
print("=====>>import data X<<===========")
print(X)

from sklearn.preprocessing import Imputer

#缺失值用NaN代替，策略用平均值， 轴心为0
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)

imputer = imputer.fit(X[:, 1:3])

X[:, 1:3] = imputer.transform(X[:, 1:3])
print("========>>process missing data X<<=========")
print(X)

#需要将数据中yes/no解析成数字值
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:,0])

print("========>>process encode<<=======")
print(X)
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
print("=========>>process encode 1<<=======")
print(X)


labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
print("=========>>X<<===============")
print(X)
print("=========>>Y<<===============")
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

print("X_train")
print X_train

print("X_test")
print X_test


print("Y_train")
print Y_train
print("Y_test")
print Y_test

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

print(X_train)
print(X_test)
