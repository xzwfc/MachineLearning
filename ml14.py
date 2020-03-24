
# based on ml13 and become a function to be used on apply_ml14
import pandas as pd 
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import numpy as np

from sklearn import metrics

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


if __name__=="__main__": #when importing this template, the lines below will be skipped

	df=pd.read_csv("ols_dataset.csv")
	target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
	data = df.iloc[:, 3:10].values
	r2_scores =run_kfold(5, data, target)
	print (r2_scores)
