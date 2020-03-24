# in terminal: head dataset.csv or head -100 dataset.csv > dataset_subset.csv (contain 100 rows)
# drop id; but merge dataset by id
# sklearn sometimes does not work in pandas dataframe, so we don't use dataframe when using skitlearn, but use numpy.
# sklearn does not accpet missing values
#any supervised learning could be embeded into this template
# in terms of kfold, when there are two survey samples, we could do two kfolds in each sample.

import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

from sklearn import metrics


df=pd.read_csv("ols_dataset.csv")
# print(df)
target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
data = df.iloc[:, 3:10].values

kfold_object=KFold(n_splits =4) #KFold only uses each value once, so we separate the dataset into 4
kfold_object.get_n_splits(data)

for training_index, test_index in kfold_object.split(data):
	print(training_index)
	print(test_index)
	data_training, data_test=data[training_index], data[test_index]
	target_training, target_test=target[training_index], target[test_index]
	machine=linear_model.LinearRegression()
	machine.fit (data_training, target_training)
	prediction =machine.predict(data_test)
	print(metrics.r2_score(target_test, prediction)) #r-square is the distance between test and prediction, the higher, the model is predicting more correctly

# -----------------------------------------------------

def run_kfold(split_number, data, target):
	# pass
	kfold_object=KFold(n_splits =split_number) #KFold only uses each value once, so we separate the dataset into 4
	kfold_object.get_n_splits(data)

	results =[]
	for training_index, test_index in kfold_object.split(data):

		print(training_index)
		print(test_index)
		data_training, data_test=data[training_index], data[test_index]
		target_training, target_test=target[training_index], target[test_index]
		machine=linear_model.LinearRegression()
		machine.fit (data_training, target_training)
		prediction =machine.predict(data_test)
		results.append(metrics.r2_score(target_test, prediction))
	return results


df=pd.read_csv("ols_dataset.csv")
# print(df)
target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
data = df.iloc[:, 3:10].values
r2_scores =run_kfold(3, data, target)
print (r2_scores)

