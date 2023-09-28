import time
from collections import Counter, OrderedDict

import pandas as pd
import sklearn.metrics
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.manifold import TSNE
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, KMeansSMOTE, SMOTENC, BorderlineSMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, label_binarize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import datetime
from typing import List
import os
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.ensemble import AdaBoostClassifier
# from imblearn.ensemble import RUSBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mord import LogisticAT
from regression import every_class_acc


def get_eval_indicator_clf(y_test, y_pre):
    '''
    :param y_test: 真实值
    :param y_pre: 预测值（模型预测出来的)
    :return: 4种评价指标
    返回回归任务的4种评价指标
    '''
    acc = accuracy_score(y_test, y_pre)
    precision = precision_score(y_test, y_pre, average='weighted')
    recall = recall_score(y_test, y_pre, average='weighted')
    f1 = f1_score(y_test, y_pre, average='weighted')

    labels = [0, 1, 2, 3, 4, 5, 6]
    y_test = label_binarize(y_test, classes=labels)
    y_pre = label_binarize(y_pre, classes=labels)

    auc = roc_auc_score(y_test, y_pre, average='weighted', multi_class='ovr')

    return acc, precision, recall, f1, auc

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
            acc, precision, recall, f1, auc = get_eval_indicator_clf(y_train, y_pre_tr)

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

            accuracy_by_class = every_class_acc(y_test, y_pre)
            class_accuracies.append(accuracy_by_class)
            # 使用Counter计算每个类别的数量
            class_counts = Counter(y_pre)
            # 按照类别从小到大排列
            sorted_class_counts = OrderedDict(sorted(class_counts.items()))
            # 打印每个类别的数量
            for class_label, count in sorted_class_counts.items():
                print(f"类别 {class_label} 的预测数量: {count}")

            # y_train_proba.append(model.predict_proba(X))
            #
            y_final_proba.append(model.predict_proba(X_test))


    final_proba = np.mean(np.array(y_final_proba), axis=0)

    # 计算总体准确率的平均值
    total_accuracies = []
    for class_label in np.unique(Y):
        class_total_accuracy = np.mean([accuracy[class_label] for accuracy in class_accuracies])
        total_accuracies.append(class_total_accuracy)

    # 打印每个类别的平均准确率
    for class_label, avg_accuracy in enumerate(total_accuracies):
        print(f"类别 {class_label} 的平均准确率: {avg_accuracy:.2f}")
    print(1)

    # df.groupby('模型').mean().to_csv('result_mean_nosmo.csv', encoding='utf-8_sig')
    #
    final_proba = pd.DataFrame(final_proba, columns=['0', '1'])
    # final_proba.to_csv('result_proba.csv', encoding='utf-8_sig')
    # df.groupby('模型').mean().to_csv('result_mean_rf_smote.csv', encoding='utf-8_sig')

    final_proba.to_csv('result_proba_svm.csv', encoding='utf-8_sig')
    df.groupby('模型').mean().to_csv('result_mean_svm_smote.csv', encoding='utf-8_sig')

def train_data_model(X, Y, X_test, models: List):
    print(X.shape)
    print(Y.shape)
    Y = Y.to_numpy(dtype=int)
    # 创建df，存放n个模型的4项指标
    # df = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2'])
    df = pd.DataFrame()
    df_index = 0  # index递增，用于存放数据

    y_train_proba = []
    y_final_proba = []
    class_accuracies = []

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

    all_models=[KNeighborsClassifier(),MLPClassifier(),SVC(probability=True, decision_function_shape='ovo'),
                RandomForestClassifier(n_estimators=100),
                XGBClassifier(n_estimators=100, max_depth=3,),
                LogisticAT()]
    all_models_name = ['KNN', 'MLP', 'SVM', 'RF', 'XGBoost','LogisticAT']  # 模型名字，方便画图


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
    skf = RepeatedStratifiedKFold(n_repeats=5, n_splits=4, random_state=17)

    for train_index, test_index in skf.split(X, Y):
        # 获取训练集与测试集
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        num = 20
        smo = BorderlineSMOTE(random_state=42, kind="borderline-1", k_neighbors=2,
                              sampling_strategy={0: num,
                                                 1: num,
                                                 2: num,
                                                 3: num,
                                                 4: num,
                                                 5: num,
                                                 6: num},
                              )
        # smo = SMOTENC(random_state=42, k_neighbors=2,
        #               sampling_strategy={0: num,
        #                                  1: num,
        #                                  2: num,
        #                                  3: num,
        #                                  4: num,
        #                                  5: num,
        #                                  6: num},
        #               categorical_features=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20])
        # x_train, y_train = smo.fit_resample(x_train, y_train)

        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)


        for model in models:
            model_index = all_models_name.index(model)
            model_name = all_models_name[model_index]
            model = all_models[model_index]
            print(f"当前模型{model_name}")

            if model_name == 'XGBoost':
                eval_set = [(x_train, y_train), (x_test, y_test)]
                model.fit(x_train, y_train, eval_set=eval_set, eval_metric='auc',verbose=True,early_stopping_rounds=5)
            # elif model_name == 'RF':
            #     model.fit(x_train, y_train)
            #     importances = model.feature_importances_
            #     indices = np.argsort(importances)[::-1]
            #     for f in range(x_train.shape[1]):
            #         print("%2d) %-*s %f" % (f + 1, 30, X[indices[f]], importances[indices[f]]))
            else:
                model.fit(x_train, y_train)


            # 获取训练集评价指标
            y_pre_tr = model.predict(x_train)
            acc, precision, recall, f1, auc = get_eval_indicator_clf(y_train, y_pre_tr)

            df.loc[df_index, '模型'] = model_name
            df.loc[df_index, 'acc_t'] = acc
            df.loc[df_index, 'f1_t'] = f1
            df.loc[df_index, 'auc_T'] = auc
            print(f"train acc: {acc}")
            print(f"train f1: {f1}")
            print(f"train auc: {auc}")

            # 获取测试集的评价指标
            y_pre = model.predict(x_test)
            acc, precision, recall, f1, auc = get_eval_indicator_clf(y_test, y_pre)

            df.loc[df_index, 'acc'] = acc
            df.loc[df_index, 'precision'] = precision
            df.loc[df_index, 'recall'] = recall
            df.loc[df_index, 'f1'] = f1
            df.loc[df_index, 'auc'] = auc
            print(f"test acc: {acc}")
            print(f"test f1: {f1}")
            print(f"test auc: {auc}")
            model_index += 1
            df_index += 1

            accuracy_by_class = every_class_acc(y_test, y_pre)
            class_accuracies.append(accuracy_by_class)
            # 使用Counter计算每个类别的数量
            class_counts = Counter(y_pre)
            # 按照类别从小到大排列
            sorted_class_counts = OrderedDict(sorted(class_counts.items()))
            # 打印每个类别的数量
            for class_label, count in sorted_class_counts.items():
                print(f"类别 {class_label} 的预测数量: {count}")

            # y_train_proba.append(model.predict_proba(X))
            if len(models) == 1:
                y_final_proba.append(model.predict_proba(X_test))

    # 计算总体准确率的平均值
    total_accuracies = []
    for class_label in np.unique(Y):
        class_total_accuracy = np.mean([accuracy[class_label] for accuracy in class_accuracies])
        total_accuracies.append(class_total_accuracy)

    # 打印每个类别的平均准确率
    for class_label, avg_accuracy in enumerate(total_accuracies):
        print(f"类别 {class_label} 的平均准确率: {avg_accuracy:.2f}")
    if len(models) == 1:
        final_proba = np.mean(np.array(y_final_proba), axis=0)
        final_proba = pd.DataFrame(final_proba, columns=['0', '1', '2', '3', '4', '5', '6'])
        final_proba.to_csv(f'result_proba_{models[0]}.csv', encoding='utf-8_sig')
        df.groupby('模型').mean().to_csv(f'result_mean_{models[0]}_smote.csv', encoding='utf-8_sig')
    else:
        df.groupby('模型').mean().to_csv('result_mean_nosmo.csv', encoding='utf-8_sig')
    print(1)


if __name__ == '__main__':
    # 获取自变量
    # X = pd.read_excel('dataset/Molecular_Descriptor.xlsx', index_col=[0], sheet_name='training')
    # X_test = pd.read_excel('dataset/Molecular_Descriptor.xlsx', index_col=[0], sheet_name='test')

    data = pd.read_csv('multiclass_label.csv', index_col=False)
    data = data.drop(['流水号'], axis=1)
    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]

    # data = pd.read_csv('multiclass_label_66feat.csv', index_col=False)
    # X = data

    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    # 创建PCA模型并指定要降到的维度
    pca = PCA(n_components='mle')
    X_pca = pca.fit_transform(X)

    X_train = X.iloc[:100, :]
    X_test = X
    Y_train = Y.iloc[:100]
    Y_test = Y.iloc[100:]

    # train_data_model(X_train, Y_train, X_test, ['RF', 'MLP', 'XGBoost', 'SVM', 'LogisticAT'])
    # train_data_model(X_train, Y_train, X_test, ['RF'])
    # train_data_model(X_train, Y_train, X_test, ['MLP'])
    # train_data_model(X_train, Y_train, X_test, ['XGBoost'])
    # train_data_model(X_train, Y_train, X_test, ['SVM'])
    train_data_model(X_train, Y_train, X_test, ['LogisticAT'])

