import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('../data/表2-患者影像信息血肿及水肿的体积及位置.csv', index_col=False)

df2 = df.iloc[:, 2:24]


# 使用 MinMaxScaler 对数据进行归一化
scaler = MinMaxScaler()
normalized_df = pd.DataFrame(scaler.fit_transform(df2), columns=df2.columns)


def dist(sub_num, normalized_df):
    normalized_df = normalized_df.values
    target_row = normalized_df[sub_num-1]
    # 计算每行与已知数据的距离，并存储在一个列表中
    distances = []
    for i, row in enumerate(normalized_df):
        distance = np.linalg.norm(row - target_row)  # 使用欧几里得距离计算
        distances.append(distance)
    # 找到最小距离对应的行
    nearest_neighbor_index = np.argmin(distances)
    # 将第一个最小距离设置为一个较大的值，以便找到第二小的距离
    distances[nearest_neighbor_index] = np.inf
    # 找到第二小距离对应的行
    second_nearest_neighbor_index = np.argmin(distances)


    print(f"{sub_num}最近邻的人为：sub{second_nearest_neighbor_index+1}")
    print(df2.loc[second_nearest_neighbor_index])

dist(131, normalized_df)
dist(132, normalized_df)




