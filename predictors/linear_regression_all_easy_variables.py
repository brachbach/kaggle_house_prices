# predict test based on linear regression on all numerical, continuous variables

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from pandas import DataFrame
import numpy as np

original_features = [
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
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

all_numerical_features = [
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

def prepare_x(house_data, params={}):
    scale = params['scale'] if 'scale' in params else True
    features = params['features'] if 'features' in params else all_numerical_features
    x = house_data.loc[ : , features]
    cleaned_x = x.fillna(x.mean())
  
    if not scale:
        return cleaned_x

    scaled_x = (cleaned_x-cleaned_x.min())/(cleaned_x.max()-cleaned_x.min())
    return scaled_x

def predict_linear(train_x, train_y, test_x, reg, clip):
    reg.fit(train_x, train_y)

    naive_price_predictions = reg.predict(test_x)

    clipped = np.array(naive_price_predictions).clip(clip)

    return clipped

def predict_with_all_easy_linear(train, test, params={}):
    train_x = prepare_x(train, params)
    train_y = train.loc[ : , "SalePrice"]
    test_x = prepare_x(test, params)

    reg = linear_model.LinearRegression()

    predictions = predict_linear(train_x, train_y, test_x, reg, 100000)

    return test.assign(SalePrice=predictions)

def predict_with_ridge(train, test, params={}):
    train_x = prepare_x(train, params)
    train_y = train.loc[ : , "SalePrice"]
    test_x = prepare_x(test, params)

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    predictions = predict_linear(train_x, train_y, test_x, reg, 100000)

    return test.assign(SalePrice=predictions)

def predict_with_polynomial_features_and_ridge(train, test, params):
    poly = PolynomialFeatures(params['degree'])

    train_x = poly.fit_transform(prepare_x(train, params))
    train_y = train.loc[ : , "SalePrice"]
    test_x = poly.fit_transform(prepare_x(test, params))

    reg = linear_model.RidgeCV(alphas=np.logspace(-6, 6, 13))

    predictions = predict_linear(train_x, train_y, test_x, reg, params['clip'])

    return test.assign(SalePrice=predictions)