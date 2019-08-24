# just predict that ever house in test will sell for the mean value from train

import csv

with open('data_from_kaggle/train.csv', 'r') as file:
    reader = csv.reader(file)
    as_list = list(reader)

print(as_list)