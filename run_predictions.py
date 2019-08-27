from predict_prices import predict_prices, evalute_predictor
from predictors.median import predict_with_median
from predictors.linear_regression_one_variable import predict_with_lot_area
from predictors.linear_regression_all_easy_variables import predict_with_all_easy_linear, predict_with_ridge, predict_with_polynomial_features_and_ridge

predict_prices("data_from_kaggle/train.csv", "data_from_kaggle/test.csv", predict_with_median)
predict_prices("data_from_kaggle/train.csv", "data_from_kaggle/test.csv", predict_with_lot_area)
predict_prices("data_from_kaggle/train.csv", "data_from_kaggle/test.csv", predict_with_all_easy_linear)
predict_prices("data_from_kaggle/train.csv", "data_from_kaggle/test.csv", predict_with_ridge)
predict_prices("data_from_kaggle/train.csv", "data_from_kaggle/test.csv", predict_with_polynomial_features_and_ridge, {"degree": 2, "clip": 100000})
