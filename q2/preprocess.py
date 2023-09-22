import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime,timedelta


def get_ed_time():
    df = pd.read_csv("data/origin/表2-患者影像信息血肿及水肿的体积及位置.csv",index_col=False)
    df_time = pd.read_csv("data/origin/time.csv")

    cols = []
    for c in df.columns:
        if c.find('时间') != -1 or c.find('ED_volume') != -1:
            cols.append(c)
    df = df[cols]
    print(cols)

    for index in df.index:
        first_check = datetime.strptime(df.loc[index, '入院首次检查时间点'], '%Y/%m/%d %H:%M')
        first_time = df.loc[index,'发病到首次影像检查时间间隔']
        for c in df.columns:
            if c.find('随访') != -1:


                time = df.loc[index,c]


                if pd.isnull(time):
                    continue
                time_difference = datetime.strptime(time, '%Y/%m/%d %H:%M') - first_check
                sec = time_difference.total_seconds()
                df.loc[index,c]=round(sec/3600+first_time,2)
    df.iloc[:,1]= df.iloc[:,0]
    df.iloc[:,0]=0
    print(df.head())

    df.to_csv("data/ed_volume_time.csv",encoding='utf-8_sig')

def get_train_data():
    df = pd.read_csv("data/ed_volume_time.csv",index_col=False)
    df = df.loc[:,~df.columns.str.contains('Unnamed')]

    print(df.info())

    X = df.loc[:,df.columns.str.contains('时间点')]
    Y = df.loc[:,df.columns.str.contains('volume')]

    print(X)
    print(Y)

    for index in X.index[:50]:
        plt.scatter(X.loc[index,:],Y.loc[index,:])
        plt.plot(X.loc[index,:],Y.loc[index,:])
    plt.show()

get_train_data()

