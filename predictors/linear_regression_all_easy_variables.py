# predict test based on linear regression on all numerical, continuous variables

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

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

def predict_with_all_easy_linear(train, test):

    # commented-out features are known or strongly suspected to contain at least one non-numerical value

    linearRegressor = linear_model.LinearRegression()

    linearRegressor.fit(train.loc[ : , easy_numerical_features], train.loc[ : , "SalePrice"])

    naive_price_predictions = linearRegressor.predict(test.loc[ : , easy_numerical_features])

    no_zeros = np.array(naive_price_predictions).clip(100000)

    output = test.assign(SalePrice=no_zeros)

    return output

def predict_with_ridge(train, test):

    # commented-out features are known or strongly suspected to contain at least one non-numerical value

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    reg.fit(train.loc[ : , easy_numerical_features], train.loc[ : , "SalePrice"])

    naive_price_predictions = reg.predict(test.loc[ : , easy_numerical_features])

    clipped = np.array(naive_price_predictions).clip(100000)

    output = test.assign(SalePrice=clipped)

    return output

def predict_with_polynomial_features_and_ridge(train, test):

    # commented-out features are known or strongly suspected to contain at least one non-numerical value

    poly = PolynomialFeatures(2)

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    reg.fit(poly.fit_transform(train.loc[ : , easy_numerical_features]), train.loc[ : , "SalePrice"])

    naive_price_predictions = reg.predict(poly.fit_transform(test.loc[ : , easy_numerical_features]))

    clipped = np.array(naive_price_predictions).clip(100000)

    output = test.assign(SalePrice=clipped)

    return output
