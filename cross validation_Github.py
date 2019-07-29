# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 20:34:38 2018

@author: LW
"""

# Basic packages
import numpy as np
import pandas as pd
from sklearn import preprocessing   # module used for normalization
from sklearn.learning_curve import validation_curve
import matplotlib.pyplot as plt

# The machine learning modules
from sklearn import svm

df = pd.read_csv("DOE(GSD)_1stRUN.csv")
data = df[['Cu_content','Temperature','Time','EDTA']]
target = df[['FoM']]
X = data
y = target

#test C value
param_range = np.linspace(1,101,50)   # set the range for parameter C
train_loss, test_loss = validation_curve(
        svm.SVR(kernel='rbf', gamma=0.5), 
        preprocessing.scale(X), preprocessing.scale(y.values.ravel()), param_name='C',
        param_range=param_range, cv=10, 
        scoring = 'neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)
# use minus to avoid negatives

# make the learning curve for C
plt.figure(1)
plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross_validation")
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
font = {'size': 22}
plt.xlabel("C", font)
plt.ylabel("Loss", font)
plt.legend(loc="best", fontsize=22)
plt.style.use('seaborn')
plt.show()

#test gamma value
param_range = np.linspace(0,1,50)   # set the range for parameter gamma
train_loss, test_loss = validation_curve( 
        svm.SVR(kernel='rbf', C=40),
        preprocessing.scale(X), preprocessing.scale(y.values.ravel()), param_name='gamma',
        param_range=param_range, cv=10, 
        scoring = 'neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

# make the learning curve for gamma
plt.figure(2)
plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross_validation")
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
font = {'size': 22}
plt.xlabel("gamma", font)
plt.ylabel("Loss", font)
plt.legend(loc="best", fontsize=22)
plt.style.use('seaborn')
plt.show()

#test epsilon value
param_range = np.linspace(0,1,50)   # set the range for parameter gamma
train_loss, test_loss = validation_curve( 
        svm.SVR(kernel='rbf', C=40, gamma=0.5),
        preprocessing.scale(X), preprocessing.scale(y.values.ravel()), param_name='epsilon',
        param_range=param_range, cv=10, 
        scoring = 'neg_mean_squared_error')
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

# make the learning curve for gamma
plt.figure(3)
plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross_validation")
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
font = {'size': 22}
plt.xlabel("epsilon", font)
plt.ylabel("Loss", font)
plt.legend(loc="best", fontsize=22)
plt.style.use('seaborn')
plt.show()