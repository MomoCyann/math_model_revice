from collections import Counter, OrderedDict

import numpy as np
from imblearn.over_sampling import SMOTENC
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize

def every_class_acc(y_test, y_pre):
    # 计算每个类别的预测准确率
    accuracy_by_class = {}
    for class_label in np.unique(y_test):
        indices = (y_test == class_label)
        accuracy = accuracy_score(y_test[indices], y_pre[indices])
        accuracy_by_class[class_label] = accuracy
    return accuracy_by_class

def get_eval_indicator_clf(y_test, y_pre):
    '''
    :param y_test: 真实值
    :param y_pre: 预测值（模型预测出来的)
    :return: 4种评价指标
    返回回归任务的4种评价指标
    '''
    acc = accuracy_score(y_test, y_pre)
    f1 = f1_score(y_test, y_pre, average='weighted')

    labels = [0, 1, 2, 3, 4, 5, 6]
    y_test = label_binarize(y_test, classes=labels)
    y_pre = label_binarize(y_pre, classes=labels)
    try:
        auc = roc_auc_score(y_test, y_pre, average='weighted', multi_class='ovr')
    except ValueError:
        auc=0
        pass

    return acc, f1, auc


def train_data_model(X, Y, X_test, models):
    print(X.shape)
    print(Y.shape)

    # 创建df，存放n个模型的4项指标
    # df = pd.DataFrame(columns=['MAE', 'MSE', 'RMSE', 'R2'])
    df = pd.DataFrame()
    df_index = 0  # index递增，用于存放数据

    y_train_proba = []
    y_final_proba = []
    # 保存每次交叉验证的类别准确率
    class_accuracies = []

    all_models=[LinearRegression()]
    all_models_name = ['Linear']  # 模型名字，方便画图

    # 交叉检验
    skf = RepeatedStratifiedKFold(n_repeats=5, n_splits=4, random_state=17)

    for train_index, test_index in skf.split(X, Y):
        # 获取训练集与测试集
        x_train, x_test = X.loc[train_index], X.loc[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        num = 20
        # smo = BorderlineSMOTE(random_state=42, kind="borderline-1", k_neighbors=2,
        #                       sampling_strategy={0: num,
        #                                          1: num,
        #                                          2: num,
        #                                          3: num,
        #                                          4: num,
        #                                          5: num,
        #                                          6: num},
        #                       )
        smo = SMOTENC(random_state=42, k_neighbors=2,
                      sampling_strategy={0: num,
                                         1: num,
                                         2: num,
                                         3: num,
                                         4: num,
                                         5: num,
                                         6: num},
                      categorical_features=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 19, 20])
        x_train, y_train = smo.fit_resample(x_train, y_train)

        print(x_train.shape, x_test.shape)
        print(y_train.shape, y_test.shape)


        for model in models:
            model_index = all_models_name.index(model)
            model_name = all_models_name[model_index]
            model = all_models[model_index]
            print(f"当前模型{model_name}")

            model.fit(x_train, y_train)


            # 获取训练集评价指标
            y_pre_tr = model.predict(x_train)
            # 四舍五入并满足条件
            y_pre_tr = np.ceil(y_pre_tr)
            y_pre_tr[y_pre_tr > 6] = 6
            y_pre_tr[y_pre_tr < 0] = 0
            y_pre_tr.astype(int)

            acc, f1, auc = get_eval_indicator_clf(y_train, y_pre_tr)

            df.loc[df_index, '模型'] = model_name
            df.loc[df_index, 'acc_t'] = acc
            df.loc[df_index, 'f1_t'] = f1
            df.loc[df_index, 'auc_T'] = auc
            print(f"train acc: {acc}")
            print(f"train f1: {f1}")
            # print(f"train auc: {auc}")

            # 获取测试集的评价指标
            y_pre = model.predict(x_test)

            # 四舍五入并满足条件
            y_pre = np.ceil(y_pre)
            y_pre[y_pre > 5] = 5
            y_pre[y_pre < 1] = 1
            y_pre.astype(int)

            acc, f1, auc = get_eval_indicator_clf(y_test, y_pre)

            df.loc[df_index, 'acc'] = acc
            df.loc[df_index, 'f1'] = f1
            df.loc[df_index, 'auc'] = auc
            print(f"test acc: {acc}")
            print(f"test f1: {f1}")
            # print(f"test auc: {auc}")
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
            # if len(models) == 1:
            #     y_final_proba.append(model.predict_proba(X_test))

    # 计算总体准确率的平均值
    total_accuracies = []
    for class_label in np.unique(Y):
        class_total_accuracy = np.mean([accuracy[class_label] for accuracy in class_accuracies])
        total_accuracies.append(class_total_accuracy)

    # 打印每个类别的平均准确率
    for class_label, avg_accuracy in enumerate(total_accuracies):
        print(f"类别 {class_label} 的平均准确率: {avg_accuracy:.2f}")


    if len(models) == 1:
        # final_proba = np.mean(np.array(y_final_proba), axis=0)
        # final_proba = pd.DataFrame(final_proba, columns=['0', '1', '2', '3', '4', '5', '6'])
        # final_proba.to_csv(f'result_proba_{models[0]}.csv', encoding='utf-8_sig')
        df.groupby('模型').mean().to_csv(f'result_mean_{models[0]}_smote.csv', encoding='utf-8_sig')
    else:
        df.groupby('模型').mean().to_csv('result_mean_nosmo.csv', encoding='utf-8_sig')
    print(1)

if __name__ == '__main__':
    data = pd.read_csv('multiclass_label.csv')
    data = data.drop(['流水号'], axis=1)
    X = data.iloc[:, 1:-1]
    Y = data.iloc[:, -1]

    data = pd.read_csv('multiclass_label_66feat.csv', index_col=False)
    X = data

    X_train = X.iloc[:100, :]
    X_test = X.iloc[100:, :]
    Y_train = Y.iloc[:100]
    Y_test = Y.iloc[100:]

    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_test = pd.DataFrame(scaler.transform(X))

    train_data_model(X_train, Y_train, X_test, ['Linear'])
