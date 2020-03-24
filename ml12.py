# in terminal: head dataset.csv or head -100 dataset.csv > dataset_subset.csv (contain 100 rows)
# drop id; but merge dataset by id
# sklearn sometimes does not work in pandas dataframe, so we don't use dataframe when using skitlearn, but use numpy.
# sklearn does not accpet missing values
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import metrics


df=pd.read_csv("ols_dataset.csv")
# print(df)
target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
data = df.iloc[:, 3:10].values

data_training, data_testing, target_training, target_testing =train_test_split(data, target, test_size =0.25, random_state=0) #propotion of test =25%
#it will retrun 4 things(data in training and testing, target in training and testing)

# print(data_training)
# print(data_testing)
# print(target_training)
# print(target_testing)

print(data_training.shape)
print(data_testing.shape)
print(target_training.shape)
print(target_testing.shape)


machine=linear_model.LinearRegression()
machine.fit(data_training, target_training)
prediction=machine.predict(data_testing)

print(prediction)

# plt.scatter(target_testing, prediction)
# plt.xlabel("Target of test dateset")
# plt.ylabel("Model prediction")
# plt.savefig("scatter_test_prediction.png")

print(metrics.r2_score(target_testing, prediction)) #R-square of the machine

