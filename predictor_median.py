# just predict that ever house in test will sell for the median value from train

def predict_with_median(train, test):
    median = train.loc[ : , "SalePrice"].median()
    test["SalePrice"] = median
    return test