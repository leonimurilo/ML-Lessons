import numpy as np #math tools
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# turn text into number with labeling
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# create one column for each label on the categorical colum
onehotencoder = OneHotEncoder(categorical_features=[3])

X = onehotencoder.fit_transform(X).toarray()

# Avoiding dummy variables trap (removing one of the dummy columns)
X = X[:, 1:]

# spliting the dataset into training set and test set:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fit model into the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting test set results
y_pred = regressor.predict(X_test)

## see the difference between the prediction and the test set real result (y)
#real_vs_pred = np.absolute(np.subtract(y_pred, y_test))
#
#plt.scatter(real_vs_pred, np.arange(10), color = 'red')
#plt.title('Precision chart (closer to 0 is better)')
#plt.xlabel('predict vs real difference')
#plt.ylabel('observation')
#plt.show()

# ============================
# building the optimal model using Backward Elimination 
# Defined SL = 0.5
# (finding out which feature is really statistically significant to the independent variable)
import statsmodels.formula.api as sm
# adds a column with 1's to represent the b0 constant for statsmodels to work properly
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # fit the model with all possible predictors (step 2)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# after removing x2 (the predictor with the highest p-value (step 3))
X_opt = X[:, [0, 1, 3, 4, 5]] # fit the model with all possible predictors (step 2)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# after removing x1 (the predictor with the highest p-value (step 3))
X_opt = X[:, [0, 3, 4, 5]] # fit the model with all possible predictors (step 2)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]] # fit the model with all possible predictors (step 2)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]] # fit the model with all possible predictors (step 2)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# in this case, the optimal team of independent varibles that 
# can predict the profit with the highest statistical significance
# is composed only by the R&D spend variable.

# R-square and adjusted R-square will be used in the next lesson
# with them we will be able to decide whether we want to keep or not 
# some dependent variables


