# predict test based on a simple linear regression on train

import pandas as pd

from sklearn.linear_model import LinearRegression

train = pd.read_csv("data_from_kaggle/train.csv")
test = pd.read_csv("data_from_kaggle/test.csv")

linearRegressor = LinearRegression()

linearRegressor.fit(train.loc[ : , "LotArea"].values.reshape(-1,1), train.loc[ : , "SalePrice"])

price_predictions = linearRegressor.predict(test.loc[ : , "LotArea"].values.reshape(-1,1))

test["SalePrice"] = price_predictions

# test["SalePrice"] = median

test.to_csv(path_or_buf="predictions/linear_regression_one_variable.csv", columns=["Id", "SalePrice"], index=False)