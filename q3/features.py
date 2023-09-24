import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import dcor


def ShowHeatMap(DataFrame, title):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    colormap = plt.cm.RdBu
    plt.figure(figsize=(14, 12))
    plt.title(title, y=1.05, size=15)
    sns.heatmap(DataFrame.astype(float), square=True, cmap="YlGnBu",annot=False)
    plt.tick_params(labelsize=12, left=False, bottom=False)
    plt.yticks(rotation=0)
    plt.xticks(rotation=45)
    plt.show()

def dcor_all_features(df):
    '''
    这个是返回变量间的关系。
    :param df:
    :return:
    '''
    def dcor_matrix(dataframe):

        data = np.array(dataframe)
        n = len(data[0, :])
        result = np.zeros([n, n])

        for i in range(n):
            for j in range(n):
                d_cor = dcor.distance_correlation(data[:, i], data[:, j])
                result[i, j] = d_cor
                result[j, i] = d_cor
        RT = pd.DataFrame(result)
        return RT

    data_dcor = dcor_matrix(df)
    data_dcor.columns = df.columns
    data_dcor['column'] = df.columns
    data_dcor.set_index('column', inplace=True)
    data_dcor.to_csv('dcor_all_features.csv')

    ShowHeatMap(data_dcor, 'DCOR')

def check(features):
    # 相关性检验
    # 距离相关系数
    # dcor_all_features(features)
    # 皮尔逊
    matrix = features.corr()
    plt.figure(figsize=(12, 12))
    sns.heatmap(features.corr(), square=True, annot=False)
    plt.tick_params(labelsize=12, left=False, bottom=False)
    plt.xticks(rotation=45)
    plt.show()
    # 设置相关系数的阈值，这里假设大于0.7的相关系数是较大的
    threshold = 0.8
    # 创建一个集合(set)用于存储要删除的列名
    variables_to_delete = []

    # 遍历相关系数矩阵，找出相关性较大的变量对，并标记要删除的列名
    for i in range(len(matrix.columns)):
        for j in range(i + 1, len(matrix.columns)):
            if abs(matrix.iloc[i, j]) > threshold:
                if matrix.columns[i] not in variables_to_delete and matrix.columns[j] not in variables_to_delete:
                    variables_to_delete.append(matrix.columns[j])
    # 删除相关性较大的列
    features_ = features.drop(columns=variables_to_delete)

    # 打印结果
    print("删除相关性较大的列后的DataFrame:")
    print(features_.shape)
    features_.to_csv('multiclass_label_66feat.csv', encoding='utf-8_sig')

def feature_selection(data, y):
    # 特征选择
    # 最大信息系数top
    mic_top_m(data, y, 40)
    # 灰色关联top
    grey_top_m(data, y, 40)
    # dcor top
    dcor_top_m(data, y, 40)
    # rf and permutation_importance
    rf_features(data, y, 40)

if __name__ == '__main__':
    data = pd.read_csv('multiclass_label.csv')
    data = data.drop(['流水号'], axis=1)
    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]

    check(X)
    # feature_selection(X, Y)

