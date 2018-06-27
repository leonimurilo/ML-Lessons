# Data Preprocessing Template

# Importing the libraries
import numpy as np # numpy contains mathematical tools
import matplotlib.pyplot as plt # matplotlib is used to plot nice charts
import pandas as pd # used to import and manager datasets

# X is the matrix of features
# y is the dependent variables vector

#importing the dataset
dataset = pd.read_csv('Data.csv')
# [:, :-1] takes all the rows and all the columns except the last one
X = dataset.iloc[:, :-1].values
# all lines and the last column
y = dataset.iloc[:, 3].values

# taking care of missing data
# in this case we will put the mean of the column in the missing spots

from sklearn.preprocessing import Imputer # contain libraries to make ML models
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
# all lines and column from 1 to 2
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# spliting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# test_size is a number from 0 to 1 that defines the percentage of your
# dataset that will be used as the test set
# usually the test set size is something between 0.2 and 0.3
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
