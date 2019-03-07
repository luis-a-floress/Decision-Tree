#import os

import pandas as pd

#path = os.path.join("titanic", "train.csv")
data_set = pd.read_csv("./../data/train.csv",sep=',')
print("HEADER")
print(data_set.head())
#print(data_set.shape())
print("INFO")
print(data_set.info())
print("DESCRIPTION")
print(data_set.describe())

total = data_set.isnull().sum().sort_values(ascending=False)
percent_1 = data_set.isnull().sum()/data_set.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head(5)
print(missing_data.head())
