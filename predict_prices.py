import math

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def predict_prices(train_filepath: str, test_filepath: str, predictor, params={}):
    train = pd.read_csv(train_filepath)
    test = pd.read_csv(test_filepath)
    predictions = predictor(train, test, params) if params else predictor(train, test)
    predictions.to_csv(path_or_buf=f"predictions/{predictor.__name__}.csv", columns=["Id", "SalePrice"], index=False)

def get_validation_data(train):
    return train_test_split(train, train_size = 0.8)

def evalute_predictor(train_filepath: str, predictor, params={}):
    train = pd.read_csv(train_filepath)
    (training, validation) = get_validation_data(train)
    # print("train:", train)
    # print("training:", training)
    # print("validation:", validation)
    predictions = predictor(training, validation, params) if params else predictor(training, validation)
    validation_true_logs = np.log(validation["SalePrice"])
    validation_predicted_logs = np.log(predictions["SalePrice"])
    mse = mean_squared_error(validation_true_logs, validation_predicted_logs)
    return math.sqrt(mse)
    
    