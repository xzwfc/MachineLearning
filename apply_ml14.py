# mv ml15.py apply_ml14.py unix change file name
import ml14
import pandas as pd
df=pd.read_csv("ols_dataset.csv")
# print(df)
target = df.iloc[:,2].values #doesnot work we need to change it into arrays by adding .valuesS
data = df.iloc[:, 3:10].values
r2_scores =ml14.run_kfold(4, data, target)
print (r2_scores)