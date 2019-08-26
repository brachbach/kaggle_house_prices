# predict test based on a linear regression on train LotArea alone

import pandas as pd

from sklearn.linear_model import LinearRegression

def predict_with_lot_area(train, test):
    linearRegressor = LinearRegression()
    linearRegressor.fit(train.loc[ : , "LotArea"].values.reshape(-1,1), train.loc[ : , "SalePrice"])
    price_predictions = linearRegressor.predict(test.loc[ : , "LotArea"].values.reshape(-1,1))
    output = test.assign(SalePrice=price_predictions)
    return output