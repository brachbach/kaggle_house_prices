# just predict that ever house in test will sell for the median value from train

import pandas as pd

train = pd.read_csv("data_from_kaggle/train.csv")

median = train.loc[ : , "SalePrice"].median()

test = pd.read_csv("data_from_kaggle/test.csv")

test["SalePrice"] = median

test.to_csv(path_or_buf="predictions/median.csv", columns=["Id", "SalePrice"], index=False)