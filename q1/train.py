import time
import pandas as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import datetime

import os
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import AdaBoostClassifier
from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def get_eval_indicator(y_test, y_pre):
    '''
    :param y_test: 真实值
    :param y_pre: 预测值（模型预测出来的)
    :return: 4种评价指标
    返回回归任务的4种评价指标
    '''
    mae = mean_absolute_error(y_test, y_pre)
    mse = mean_squared_error(y_test, y_pre)
    rmse = np.sqrt(mean_squared_error(y_test, y_pre))
    r2 = r2_score(y_test, y_pre)
    return mae, mse, rmse, r2

def get_eval_indicator_clf(y_test, y_pre):
    '''
    :param y_test: 真实值
    :param y_pre: 预测值（模型预测出来的)
    :return: 4种评价指标
    返回回归任务的4种评价指标
    '''
    acc = accuracy_score(y_test, y_pre)
    f1 = f1_score(y_test, y_pre)
    auc = roc_auc_score(y_test, y_pre)
    return acc, f1, auc

def plot_box_indicator(df):
    '''
    :param df: 各模型评价指标
    :return:   4个评价指标的箱线图
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    sns.set_theme(style="whitegrid")

    df = df.loc[:, ~df.columns.str.contains('Unnamed')]
    # 对所有指标进行遍历画图
    for column in df.columns[:1]:
        ax = sns.boxplot(x="LSTM模型参数", y=column, data=df, hue='LSTM模型参数', dodge=False,
                    showmeans=True,
                    meanprops={"marker": "d",
                               "markerfacecolor": "white",
                               "markeredgecolor": "black",},
                    palette=sns.diverging_palette(240, 10, sep=12))
        model_labels = ['KNN', '多层感知机', '随机森林回归', '支持向量机回归', 'XGBoost']

        n = 0
        for i in model_labels:
            ax.legend_.texts[n].set_text(i)
            n += 1

        plt.show()


def train_data(X, Y, X_test):
    print(X.shape)
    print(Y.shape)

    # 创建df，存放n个模型的4项指标
    # df = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2'])
    df = pd.DataFrame()
    df_index = 0  # index递增，用于存放数据

    y_train_proba = []
    y_final_proba = []

    # 模型库，每次分出训练集与测试集，均在该模型库中遍历训练
    # models = [KNeighborsRegressor(weights='distance', n_neighbors=7, algorithm='kd_tree'),
    #           MLPRegressor(solver='sgd', max_iter=1700, learning_rate_init=0.001, hidden_layer_sizes=(256,),
    #                        batch_size=128, alpha=0.0001, activation='tanh',),
    #           SVR(kernel='rbf', C=0.7),
    #           RandomForestRegressor(n_estimators=1000, max_depth=20, min_samples_leaf=1),
    #           XGBRegressor(n_estimators=600, max_depth=3, gamma=0.2, min_child_weight=4,
    #                            subsample=0.7, colsample_bytree=0.8,
    #                            reg_alpha=0.05, reg_lambda=0.1)]

    # # 构建决策树分类器
    # tree = DecisionTreeClassifier()
    # # 构建RUSBoost分类器
    # rusboost = RUSBoostClassifier(base_estimator=tree, n_estimators=50, learning_rate=1.0)
    # adaboost = AdaBoostClassifier(base_estimator=tree, n_estimators=50, learning_rate=1.0)

    models=[KNeighborsClassifier(),MLPClassifier(),SVC(),RandomForestClassifier(n_estimators=300),XGBClassifier(n_estimators=300)]
    models_name = ['KNN', 'MLP', 'SVM', 'RF', 'XGBoost',]  # 模型名字，方便画图
    models=[RandomForestClassifier(n_estimators=300)]
    models_name = ['RF']
    models = [SVC(probability=True)]
    models_name = ['SVM']
    # # 测试模型
    # models = [XGBRegressor(n_estimators=600, max_depth=3, gamma=0.2, min_child_weight=4,
    #                        subsample=0.7, colsample_bytree=0.8,
    #                        reg_alpha=0.05, reg_lambda=0.1),
    #           XGBRegressor(),
    #           RandomForestRegressor(n_estimators=1000, max_depth=25, min_samples_leaf=1),
    #           RandomForestRegressor()
    #           ]
    # models_name = ['XGBoost', 'XGBoost_o', 'RF', 'RF_o']

    # 交叉检验
    skf = RepeatedStratifiedKFold(n_repeats=5, n_splits=5, random_state=17)

    for train_index, test_index in skf.split(X, Y):
        # 获取训练集与测试集
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)

        model_index = 0
        for model in models:
            model_name = models_name[model_index]
            print(f"当前模型{model_name}")

            if model_name == 'XGBoost':
                eval_set = [(x_train, y_train), (x_test, y_test)]
                model.fit(x_train, y_train, eval_set=eval_set, eval_metric='auc',verbose=True,early_stopping_rounds=5)

                # results = LSTM模型参数.evals_result()
                # plt.plot(results['validation_0']['mae'], label='train')
                # plt.plot(results['validation_1']['mae'], label='test')
                # plt.legend()
                # plt.show()

            else:
                model.fit(x_train, y_train)


            # 获取训练集评价指标
            y_pre_tr = model.predict(x_train)
            acc, f1, auc = get_eval_indicator_clf(y_train, y_pre_tr)

            # df.loc[df_index, '模型'] = model_name
            # df.loc[df_index, 'MAE_t'] = mae
            # df.loc[df_index, 'MSE_t'] = mse
            # df.loc[df_index, 'RMSE_T'] = rmse
            # df.loc[df_index, 'R2_t'] = r2
            # print(f"train+{r2}")

            df.loc[df_index, '模型'] = model_name
            df.loc[df_index, 'acc_t'] = acc
            df.loc[df_index, 'f1_t'] = f1
            df.loc[df_index, 'auc_T'] = auc
            print(f"train acc: {acc}")
            print(f"train f1: {f1}")
            print(f"train auc: {auc}")
            # 获取测试集的评价指标
            y_pre = model.predict(x_test)
            # mae, mse, rmse, r2 = get_eval_indicator(y_test, y_pre)
            acc, f1, auc = get_eval_indicator_clf(y_test, y_pre)

            df.loc[df_index, 'acc'] = acc
            df.loc[df_index, 'f1'] = f1
            df.loc[df_index, 'auc'] = auc
            print(f"test acc: {acc}")
            print(f"test f1: {f1}")
            print(f"test auc: {auc}")
            model_index += 1
            df_index += 1

            # y_train_proba.append(model.predict_proba(X))
            #
            y_final_proba.append(model.predict_proba(X_test))


    final_proba = np.mean(np.array(y_final_proba), axis=0)

    print(1)

    # df.groupby('模型').mean().to_csv('result_mean.csv', encoding='utf-8_sig')
    #
    final_proba = pd.DataFrame(final_proba, columns=['0', '1'])
    # final_proba.to_csv('result_proba.csv', encoding='utf-8_sig')
    # df.groupby('模型').mean().to_csv('result_mean_rf_smote.csv', encoding='utf-8_sig')

    final_proba.to_csv('result_proba_svm.csv', encoding='utf-8_sig')
    df.groupby('模型').mean().to_csv('result_mean_svm_smote.csv', encoding='utf-8_sig')



if __name__ == '__main__':
    # 获取自变量
    # X = pd.read_excel('dataset/Molecular_Descriptor.xlsx', index_col=[0], sheet_name='training')
    # X_test = pd.read_excel('dataset/Molecular_Descriptor.xlsx', index_col=[0], sheet_name='test')

    data = pd.read_csv('dataset_label.csv')
    data = data.drop(['流水号'], axis=1)
    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]



    X_train = X.iloc[:100, :]
    X_test = X.iloc[100:, :]
    Y_train = Y.iloc[:100]
    Y_test = Y.iloc[100:]

    smo = SMOTE(random_state=42)
    X_train, Y_train = smo.fit_resample(X_train, Y_train)


    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X))



    # 因变量归一化(如有必要)
    # scaler = preprocessing.MinMaxScaler()
    # Y = pd.DataFrame(columns=Y.columns, data=scaler.fit_transform(Y))

    # 训练并画图
    # train_data(X, Y, X_test)
    train_data(X_train, Y_train, X_test)


