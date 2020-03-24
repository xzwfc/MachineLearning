# in terminal: head dataset.csv or head -100 dataset.csv > dataset_subset.csv (contain 100 rows)
# drop id; but merge dataset by id
# sklearn sometimes does not work in pandas dataframe, so we don't use dataframe when using skitlearn, but use numpy.
# sklearn does not accpet missing values
import pandas as pd 
from sklearn import linear_model
df=pd.read_csv("ols_dataset.csv")
print(df)
target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
data = df.iloc[:, 3:10].values

machine=linear_model.LinearRegression() #make a machine that we can fit data in to let it become smarter 
#and use the machine to predict, although the results would be the same as OLS beacuse its linear model.

machine.fit(data, target) #develop the machine by feeding more data (machine is actually a linear model)

X=[[24,55,31,3,0,7,20],[40,50,2,5,1,8,20], [3,95,37,3,1,15,17],] #predict 3 datasets (3 results)

results=machine.predict(X)
print(results)

#we want to know how good is this machine (we use one part of the data to train the machine and the other part to check how good is the machine)
#training dataset for developing the machine, testing dataset is to evaluate the machine