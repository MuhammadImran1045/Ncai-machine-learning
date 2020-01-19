
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('housing price.csv')

x= dataset.iloc[:,0:1]
y= dataset.iloc[:, 1]

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train , y_test=train_test_split(  x,y , test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)


print(regressor.predict([[1222]]))
regressor.score(x_test,y_test)

"""
Improving the accuracy of model using Regularization
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=10)),
    ('model', LinearRegression())
]

pipeline = Pipeline(steps)

pipeline.fit(x_train, y_train)

print('Training score: {}'.format(pipeline.score(x_train, y_train)))
print('Test score: {}'.format(pipeline.score(x_test, y_test)))