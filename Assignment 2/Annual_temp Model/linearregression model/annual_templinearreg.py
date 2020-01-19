
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('annual_temp.csv')

# creating file handler for 
# our example.csv file in 
# read mode 
file_handler = open("annual_temp.csv", "r") 
  
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
data.Source[data.Source == 'GCAG'] = 1
data.Source[data.Source == 'GISTEMP'] = 2

print(data)

x= data.iloc[:,0:2]
y= data.iloc[:, 2]

from sklearn.model_selection import train_test_split 
x_train , x_test , y_train , y_test=train_test_split(  x,y , test_size=0.2)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train , y_train)

print(regressor.predict([[2,1880]]))
regressor.score(x_test,y_test)