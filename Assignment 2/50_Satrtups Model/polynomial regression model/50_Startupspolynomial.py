# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 19:46:45 2020

@author: HOME
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:25:32 2020

@author: HOME
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

# creating file handler for 
# our example.csv file in 
# read mode 
file_handler = open("50_Startups.csv", "r") 
  
# creating a Pandas DataFrame 
# using read_csv function that 
# reads from a csv file. 
data = pd.read_csv(file_handler, sep = ",") 
  
# closing the file handler 
file_handler.close() 
  
# traversing through Gender  
# column of dataFrame and  
# writing values where 
# condition matches.  
data.State[data.State == 'New York'] = 1
data.State[data.State == 'California'] = 2
data.State[data.State == 'Florida'] = 3


x= data.iloc[:,0:4]
y= data.iloc[:, 4]

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train , y_test=train_test_split(  x,y , test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x , y)



# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x)
poly_reg.fit(X_poly, y)
linreg_2 = LinearRegression()
linreg_2.fit(X_poly, y)
# Predicting a new result with Linear Regression
print(regressor.predict([[165349,136898,471784,1]]))


# Predicting a new result with Polynomial Regression
print(linreg_2.predict(poly_reg.fit_transform([[165349,136898,471784,1]])))

regressor.score(x_test,y_test)
