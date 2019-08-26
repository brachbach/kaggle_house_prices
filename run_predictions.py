from predict_prices import predict_prices, evalute_predictor
from predictor_median import predict_with_median

# predict_prices("data_from_kaggle/train.csv", "data_from_kaggle/test.csv", predict_with_median)
print(evalute_predictor("data_from_kaggle/train.csv", predict_with_median))