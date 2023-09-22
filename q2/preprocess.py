import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.read_csv("data/水肿体积时序.csv")
df_time = pd.read_csv("data/origin/time.csv")

print(df.head())
print(df.info())

cols = []

# 获取流水号字段
for c in df.columns:
    if c.find("流水号") != -1:
        cols.append(c)
        # df[c]=df[c].astype(float).astype(int)


print(cols)

for index in df.index:
    for col in cols:

        # 获取流水号
        num = df.loc[index,col]
        # 获取时间字段
        time_col = col.replace('流水号','时间点')
        # 获取对应时间
        time = df_time.loc[df_time[col] == num, time_col]
        # df.loc[index,time_col]=time

        # 查找它所在的行
        target_row = df_time[df_time.eq(num).any(axis=1)]
        # 如果找到目标行，获取该行的索引
        if not target_row.empty:
            row_index = target_row.index[0]
            time_column_index = df_time.columns.get_loc(col)
            # 获取同一行的前一列的值
            if row_index >= 0:
                time = df_time.iloc[row_index, time_column_index - 1]
                print(time)
        else:
            print(f"未找到目标值{row_index} {col}")



print(df.head())

