# import pandas as pd 
# from sklearn import linear_model
# df=pd.read_csv("logistic_dataset.csv")
# print(df)
# target = df.iloc[:,1].values #doesnot work we need to change it into arrays by adding .values
# data = df.iloc[:, 3:9].values

# machine=linear_model.LogisticRegression() #make a machine that we can fit data in to let it become smarter 
# #and use the machine to predict, although the results would be the same as OLS beacuse its linear model.

# machine.fit(data, target)

# X=[[24,55,31,3,0,7],[40,50,2,5,1,8], [3,95,37,3,1,15],] #predict 3 datasets (3 results)

# results=machine.predict(X)
# print(results)

# ___________________________________

import logistics1
import pandas as pd
from sklearn import linear_model
df =pd.read_csv("logistic_dataset.csv")
target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
data = df.iloc[:, 3:9].values
r2_scores =logistics1.run_kfold(5, data, target, linear_model.LogisticRegression(),1)
print (r2_scores)