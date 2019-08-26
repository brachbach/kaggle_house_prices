# just predict that ever house in test will sell for the median value from train

def predict_with_median(train, test):
    train_median = train.loc[ : , "SalePrice"].median()
    output = test.assign(SalePrice=train_median)
    return output