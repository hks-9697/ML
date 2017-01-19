import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import model_selection
from sklearn.decomposition import PCA
from math import floor
from random import shuffle
df=pd.read_csv("voice.csv")
df.head(11)



features=['meanfreq','sd','median','Q25','Q75','skew','kurt','sfm','centroid','meanfun']


l=df.iloc[np.random.permutation(len(df))]
train_data = l[:int(floor(3167*0.8))]
test_data = l[int(floor(3167*0.8)):]
#shuffle(l[:])
x=train_data[features] 	
y=train_data['label']=="male"
x_test=test_data[features]
y_test=test_data['label']=="male"
pca = PCA(n_components=5)
pca.fit(x)
X=pca.transform(x)
X_test=pca.transform(x_test)



svc = SGDClassifier(loss="log", penalty="l1")
svc.fit(x,y)

Y_test=svc.predict(x_test)
k=0.0
j=0.0
for i in Y_test==y_test:
	if i==True:
		k+=1
	j+=1
t=0.00
t=k/j
print(t)


