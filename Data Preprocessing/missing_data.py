# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # features
y = dataset.iloc[:, 3].values # label

# Taking care of missing data
# To impute means to calculate something when you do not have exact information,
# by comparing it to something similar.
from sklearn.preprocessing import Imputer

# missing_values are the ones that will be replaced
# strategy is the method that will be used to replace those values
# in this case, they will be replaced by the mean of the axis (0 is column, 1 is row)
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# fit will "adapt" the imputer to 
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])