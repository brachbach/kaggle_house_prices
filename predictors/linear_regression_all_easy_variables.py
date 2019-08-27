# predict test based on linear regression on all numerical, continuous variables

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from pandas import DataFrame
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

def prepare_x(house_data):
    x = house_data.loc[ : , easy_numerical_features]
    cleaned_x = x.fillna(x.mean())
    # print(cleaned_x)
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
    scaled_x = (cleaned_x-cleaned_x.min())/(cleaned_x.max()-cleaned_x.min())
    # print(scaled_x)
    return scaled_x

def predict_linear(train_x, train_y, test_x, reg, clip):
    reg.fit(train_x, train_y)

    naive_price_predictions = reg.predict(test_x)

    clipped = np.array(naive_price_predictions).clip(100000)

    return clipped

def predict_with_all_easy_linear(train, test):
    train_x = prepare_x(train)
    train_y = train.loc[ : , "SalePrice"]
    test_x = prepare_x(test)

    reg = linear_model.LinearRegression()

    predictions = predict_linear(train_x, train_y, test_x, reg, 100000)

    return test.assign(SalePrice=predictions)

def predict_with_ridge(train, test):
    train_x = prepare_x(train)
    train_y = train.loc[ : , "SalePrice"]
    test_x = prepare_x(test)

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    predictions = predict_linear(train_x, train_y, test_x, reg, 100000)

    return test.assign(SalePrice=predictions)

def predict_with_polynomial_features_and_ridge(train, test, params):
    poly = PolynomialFeatures(params['degree'])

    train_x = poly.fit_transform(prepare_x(train))
    train_y = train.loc[ : , "SalePrice"]
    test_x = poly.fit_transform(prepare_x(test))

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    predictions = predict_linear(train_x, train_y, test_x, reg, 100000)

    return test.assign(SalePrice=predictions)