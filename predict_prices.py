import math

import pandas as pd

def predict_prices(train_filepath: str, test_filepath: str, predictor):
    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)
    predictions = predictor(train, test)
    test.to_csv(path_or_buf=f"predictions/{predictor.__name__}.csv", columns=["Id", "SalePrice"], index=False)

def evalute_predictor(train_filepath: str, predictor):
    train = train = pd.read_csv(train_filepath)
    train_n = len(train.index)
    split_point = math.floor(train_n * 0.7)
    training = train.loc[ 0:split_point ]
    validation = train.loc[ split_point: ]
    predictions = predictor(training, validation)
    
    