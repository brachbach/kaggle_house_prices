# predict test based on a simple linear regression

import pandas as pd

from sklearn.linear_model import LinearRegression

train = pd.read_csv("data_from_kaggle/train.csv")



test = pd.read_csv("data_from_kaggle/test.csv")

# test["SalePrice"] = median

# test.to_csv(path_or_buf="predictions/median.csv", columns=["Id", "SalePrice"], index=False)