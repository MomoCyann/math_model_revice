import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime,timedelta
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

def get_train_data():
    df = pd.read_csv("data/ed_volume_time.csv",index_col=False)
    df = df.loc[:,~df.columns.str.contains('Unnamed')]

    print(df.info())

    X = df.loc[:,df.columns.str.contains('时间点')]
    Y = df.loc[:,df.columns.str.contains('volume')]

    X.fillna(0, inplace=True)
    Y.fillna(0, inplace=True)

    print(X)
    print(Y)

    for index in X.index[:5]:
        plt.scatter(X.loc[index,:],Y.loc[index,:])
        plt.plot(X.loc[index,:],Y.loc[index,:])
    plt.show()
    return X,Y

X,Y=get_train_data()

model = LinearRegression()
# model = MLPRegressor()
model.fit(X, Y)
print(model.coef_)
print(model.intercept_)
print(model.score(X, Y))
print(model.predict(X), Y)
print("\n\n")