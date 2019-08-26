# predict test based on linear regression on all numerical, continuous variables

import pandas as pd

from sklearn.linear_model import LinearRegression

train = pd.read_csv("data_from_kaggle/train.csv")
test = pd.read_csv("data_from_kaggle/test.csv")

linearRegressor = LinearRegression()

# commented-out features are known or strongly suspected to contain at least one non-numerical value
easy_numerical_features = [
    # "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    # "MasVnrArea",
    # "BsmtFinSF1",
    # "BsmtFinSF2",
    # "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    # "BsmtFullBath",
    # "BsmtHalfBath",
    # "FullBath",
    # "HalfBath",
    # "Bedroom",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    # "GarageYrBlt",
    # "GarageCars",
    # "GarageArea",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold"
]

linearRegressor.fit(train.loc[ : , easy_numerical_features], train.loc[ : , "SalePrice"])

price_predictions = linearRegressor.predict(test.loc[ : , easy_numerical_features])

test["SalePrice"] = price_predictions

test.to_csv(path_or_buf="predictions/linear_regression_all_easy_variables.csv", columns=["Id", "SalePrice"], index=False)