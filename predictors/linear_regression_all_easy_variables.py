# predict test based on linear regression on all numerical, continuous variables

from sklearn.linear_model import LinearRegression
import numpy as np

def predict_with_all_easy_linear(train, test):

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

    linearRegressor = LinearRegression()

    linearRegressor.fit(train.loc[ : , easy_numerical_features], train.loc[ : , "SalePrice"])

    naive_price_predictions = linearRegressor.predict(test.loc[ : , easy_numerical_features])

    no_zeros = np.array(naive_price_predictions).clip(100000)

    output = test.assign(SalePrice=no_zeros)

    return output
