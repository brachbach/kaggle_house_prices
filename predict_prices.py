import math

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def predict_prices(train_filepath: str, test_filepath: str, predictor):
    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)
    predictions = predictor(train, test)
    predictions.to_csv(path_or_buf=f"predictions/{predictor.__name__}.csv", columns=["Id", "SalePrice"], index=False)

def evalute_predictor(train_filepath: str, predictor):
    train = train = pd.read_csv(train_filepath)
    train_n = len(train.index)
    split_point = math.floor(train_n * 0.7)
    training = train.loc[ 0:split_point ]
    validation = train.loc[ split_point: ]
    predictions = predictor(training, validation)
    validation_true_logs = np.log(validation["SalePrice"])
    validation_predicted_logs = np.log(predictions["SalePrice"])
    mse = mean_squared_error(validation_true_logs, validation_predicted_logs)
    return math.sqrt(mse)
    
    