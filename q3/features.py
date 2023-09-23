import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

def check(features):
    # 相关性检验
    # 距离相关系数
    dcor_all_features(features)
    # 皮尔逊
    plt.figure(figsize=(12, 12))
    sns.heatmap(features.corr(), square=True, annot=False)
    plt.tick_params(labelsize=12, left=False, bottom=False)
    plt.xticks(rotation=45)
    plt.show()

data = pd.read_csv('multiclass_label.csv')
data = data.drop(['流水号'], axis=1)
X = data.iloc[:, 1:-1]