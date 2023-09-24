import pandas as pd
from fastdtw import fastdtw
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from regression import get_all_volume

df = pd.read_csv('data/ed_volume_time.csv')
df = df.loc[:99,:]

def count_dist(bound):
    # 前3个节点
    df_volume = df.loc[:,df.columns.str.contains('volume')]
    df_volume = df_volume.iloc[:,:bound]
    df_volume.fillna(0,inplace=True)

    num_samples = 100
    sample_data = df_volume.values

    # 初始化距离矩阵
    distance_matrix = np.zeros((num_samples, num_samples))

    # 计算每对样本之间的DTW距离
    for i in range(num_samples):
        for j in range(i, num_samples):
            distance, _ = fastdtw(sample_data[i], sample_data[j])
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance  # 距离矩阵是对称的，可以只计算上三角或下三角部分
    return distance_matrix

distance_matrix=np.concatenate((count_dist(2),count_dist(3),count_dist(4),count_dist(5),count_dist(6),count_dist(7),count_dist(8)),axis=1)

for cluster in range(3,6):
    model = KMeans(cluster)
    predict= model.fit_predict(distance_matrix)
    print(predict)

    df[f"label{cluster}"] = predict

    # for i in range(cluster):
    #     df_tmp = df.loc[df[f"label{cluster}"]==i,:]
    #
    #     X = df_tmp.loc[:, df_tmp.columns.str.contains('时间点')]
    #     Y = df_tmp.loc[:, df_tmp.columns.str.contains('volume')]
    #
    #     for index in X.index:
    #         # plt.scatter(X.loc[index,:],Y.loc[index,:])
    #         plt.plot(X.loc[index, :], Y.loc[index, :], alpha=0.8)
    #     plt.show()


df.to_csv('data/ed_dtw_km.csv',encoding='utf-8_sig')

# 根据dtw聚类切分数据
df = pd.read_csv('data/ed_dtw_km.csv')
cluster=5
for i in range(cluster):
    df_tmp = df.loc[df[f"label{cluster}"]==i,:]
    df_v = get_all_volume(df_tmp)
    df_v.to_csv(f"data/cluster/ed_cluster{cluster}_{i}.csv",encoding='utf-8_sig')

