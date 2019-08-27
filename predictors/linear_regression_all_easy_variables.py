# predict test based on linear regression on all numerical, continuous variables

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

easy_numerical_features = [
    "LotFrontage",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "MasVnrArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "TotalBsmtSF",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageCars",
    "GarageArea",
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

def clean(df):
    cleaned_df = df.fillna(df.mean())
    # print(cleaned_df.loc[ : , [
    #     "GarageYrBlt",
    #     "GarageCars",
    #     "GarageArea",
    #     "BsmtFullBath",
    #     "BsmtHalfBath",
    #     "FullBath",
    #     "HalfBath",
    #     "BedroomAbvGr",
    #     "MasVnrArea",
    #     "BsmtFinSF1",
    #     "BsmtFinSF2",
    #     "TotalBsmtSF"
    # ]])
    return cleaned_df

def predict_with_all_easy_linear(train, test):

    clean_train = clean(train)
    clean_test = clean(test)

    linearRegressor = linear_model.LinearRegression()

    linearRegressor.fit(clean_train.loc[ : , easy_numerical_features], clean_train.loc[ : , "SalePrice"])

    naive_price_predictions = linearRegressor.predict(clean_test.loc[ : , easy_numerical_features])

    clipped = np.array(naive_price_predictions).clip(100000)

    output = clean_test.assign(SalePrice=clipped)

    return output

def predict_with_ridge(train, test):

    clean_train = clean(train)
    clean_test = clean(test)

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    reg.fit(clean_train.loc[ : , easy_numerical_features], clean_train.loc[ : , "SalePrice"])

    naive_price_predictions = reg.predict(clean_test.loc[ : , easy_numerical_features])

    clipped = np.array(naive_price_predictions).clip(100000)

    output = clean_test.assign(SalePrice=clipped)

    return output

def predict_with_polynomial_features_and_ridge(train, test, params):

    clean_train = clean(train)
    clean_test = clean(test)

    poly = PolynomialFeatures(params['degree'])

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    reg.fit(poly.fit_transform(clean_train.loc[ : , easy_numerical_features]), clean_train.loc[ : , "SalePrice"])

    naive_price_predictions = reg.predict(poly.fit_transform(clean_test.loc[ : , easy_numerical_features]))

    clipped = np.array(naive_price_predictions).clip(params['clip'])

    output = clean_test.assign(SalePrice=clipped)

    return output
