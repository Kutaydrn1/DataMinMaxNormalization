import pandas as pd
import numpy as np
from urllib.request import urlopen
from sklearn.preprocessing import MinMaxScaler

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

veri=pd.read_csv(url, names=["sepal_length","sepal width","petal_length","petal_width","class"])
print(veri)

X=veri.iloc[:,0:4]
#data.drop['class', axis=1]
Y=veri.iloc[:,4]
#data['class']

scaleFunc=MinMaxScaler(feature_range=(0,1))
dataScaled=scaleFunc.fit_transform(X)

dataScaled=pd.DataFrame(dataScaled,columns=X.columns)
print(dataScaled)
