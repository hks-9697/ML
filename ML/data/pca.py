import pandas as pd
from sklearn.decomposition import PCA
df=pd.read_csv("iris.csv")
print(df.head())
features=['sepal length','sepal width','petal length','petal width']

def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.

    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    #print map_to_int
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)
    return (df_mod, targets)
    

df,targets=encode_target(df,"class")
X=df[features]
Y=df['Target']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn import cross_validation

train_data = pd.DataFrame()
test_data = pd.DataFrame()

l = df.iloc[np.random.permutation(len(df))]
#divide to test and train
train_data = l[:110]
test_data = l[110:]
x=train_data[features]
y=train_data['Target']
x_test=test_data[features]
y_test=test_data['Target']
colormap = np.array(['red', 'lime', 'black'])


svc = svm.SVC(kernel='linear')
pca = PCA(n_components=2)
pca.fit(x)
x=pca.transform(x)
print(x[:])
plt.scatter(x[:][0],x[],c=colormap[Y])
plt.show()







svc.fit(x,y)

Y=svc.predict(x)

print(Y==y)
