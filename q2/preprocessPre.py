# -*- coding: utf-8 -*-

import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 第一问 获取各变量的发电量占比
def count_percent():
    df = pd.read_csv("data/2006-2020各省发电量月度数据.csv", encoding='gb18030')
    c1 = df.columns

    for f in glob.glob("data/var/*.csv"):
        print(f)

        df2 = pd.read_csv(f, encoding='gb18030')
        df_percent = df2.copy()

        df = df.loc[:, df2.columns]

        df_percent.iloc[:, 1:] = round(df2.iloc[:, 1:] / df.iloc[:, 1:], 4)
        df_percent.fillna(0, inplace=True)

        df_percent.set_index('地区', inplace=True)
        df_percent = df_percent.T

        df_percent['newindex'] = np.arange(len(df_percent) - 1, -1, -1)
        df_percent.sort_values('newindex', inplace=True, ascending=True)
        df_percent.drop('newindex', axis=1, inplace=True)

        df_percent.reset_index(drop=False, inplace=True)
        df_percent.rename(columns={"index": "月份"}, inplace=True)

        print(df_percent)
        print(df_percent.columns)
        df_percent.to_csv("data/percent/" + f.split("\\")[1].split(".")[0] + "_per.csv", encoding="utf-8_sig")


def count_corr():
    for f in glob.glob("data/percent/*.csv"):
        print(f)
        df = pd.read_csv(f)

        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        df = df.loc[:, ~df.columns.str.contains('月份')]

        # 若整列数据都为0 是无法计算出相关性的, 故认为在第一行数据设为0.01,避免全为0 的情况
        cols = df.loc[:, (df == 0).any()].columns

        for c in cols:
            df.loc[0, c] = 0.01

        corr = df.corr()

        np.fill_diagonal(corr.values, 1)
        print(corr)

        corr.to_csv("data/corr/" + f.split("\\")[1].split(".")[0] + "_corr.csv", encoding="utf-8_sig")

        df = pd.read_csv(glob.glob("data/corr/*.csv")[0])
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        for f in glob.glob("data/corr/*.csv")[1:]:
            print(f)
            df_tmp = pd.read_csv(f)
            df_tmp = df_tmp.loc[:, ~df_tmp.columns.str.contains('Unnamed')]
            print(df_tmp)
            df += df_tmp

        df = round(df / 5.0, 4)
        print(df)
        df.index = df.columns
        df.to_csv('data/corr/total/total_corr.csv', encoding="utf-8_sig")


# 画总体corr热力图
def plot_corr():
    df = pd.read_csv('data/corr/total/total_corr.csv')
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    print(df.columns)

    plt.figure(figsize=(12, 10))
    sns.heatmap(df.values, xticklabels=df.columns, yticklabels=df.columns)
    plt.show()


def km_corr():
    df = pd.read_csv('data/corr/total/total_corr.csv')
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    x = df.values
    model_pca = PCA(n_components=0.95)
    x_pca = model_pca.fit_transform(x)
    print(model_pca.explained_variance_ratio_)

    # km, 遍历不同聚类数量, 取平均相关性最高
    total_corrs = []
    for c in range(6, 7):

        model_km = KMeans(n_clusters=c)
        model_km.fit(x_pca)
        label = model_km.predict(x_pca)

        # print(label)
        x1 = []
        x2 = []
        # for i in range(31):
        #     x1.append(x_pca[i,0])
        #     x2.append(x_pca[i,1])
        # plt.scatter(x1,x2,c=label)
        # plt.show()
        corrs = []
        for l in range(c):
            df_km = df.loc[label == l, label == l]
            # print(df_km)
            corr = df_km.mean().mean()
            corrs.append(corr)
            print(",".join(df.columns[label == l].values), round(np.mean(corrs), 4))
        total_corrs.append(np.mean(corrs))
        print(c, np.mean(corrs))

    plt.plot(total_corrs)
    plt.xticks(range(0, 15), range(2, 17))
    plt.grid()
    plt.show()


def data_transposition(f):
    df = pd.read_csv(f, encoding='gb18030')
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    df.set_index('地区', inplace=True)
    df = df.T
    df['newindex'] = np.arange(len(df) - 1, -1, -1)
    df.sort_values('newindex', inplace=True, ascending=True)
    df.drop('newindex', axis=1, inplace=True)
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={"index": "月份"}, inplace=True)
    print(df.head())
    df.to_csv(f, encoding="utf-8_sig")


def region_power_structure():
    df_region = pd.read_csv('data/区域.csv')
    df_region = df_region.loc[:, ~df_region.columns.str.contains('Unnamed')]

    # 用于获取月份
    df_n = pd.read_csv("data/percent/2006-2020分省核能发电量_当期值月度数据_per.csv")

    for l in range(1, 7):
        # 构建该区域的df
        df_q = pd.DataFrame(columns=['月份', '核', '风', '水', '太', '火'], index=df_n.index)
        df_q['月份'] = df_n['月份']
        print(df_q)

        cols = df_region[df_region['区域'] == l]['地区'].values
        print(cols)

        for f in glob.glob("data/percent/*.csv"):
            q_type = f[24]
            print(q_type)

            df_one = pd.read_csv(f)
            df_one = df_one.loc[:, ~df_one.columns.str.contains('Unnamed')]
            df_one = df_one.loc[:, cols].mean(axis=1)

            print(df_q.shape[0] - df_one.shape[0], df_q.shape[0])
            df_q.loc[df_q.shape[0] - df_one.shape[0]:df_q.shape[0], q_type] = df_one.values
        print(df_q)
        df_q.fillna(0, inplace=True)
        df_q['total'] = df_q.iloc[:, 1:6].sum(axis=1)
        df_q.to_csv(f"data/region/region_{l}_power_structure.csv", encoding="utf-8_sig")


def plot_region_power_stt():
    for l in range(1, 7):
        df = pd.read_csv(f"data/region/region_{l}_power_structure.csv")
        df = df.loc[:, ~df.columns.str.contains('Unnamed')]
        print(l)

        plt.figure(figsize=(12, 6))
        for i in range(1, 6):
            plt.plot(df.iloc[:, i], label=df.columns[i])
        plt.legend()
        plt.xticks(np.arange(0, df.shape[0], 4), df.loc[df.index % 4 == 0, '月份'], rotation=60)
        plt.subplots_adjust(bottom=0.15)
        plt.show()
