import pandas as pd
import numpy as np
# data = pd.read_csv('result_proba_LogisticAT.csv', index_col=0)
data=pd.read_csv('BILSTM_PROBA.csv', index_col=False)
# 找到每一行中概率最大的列名
max_prob_columns = np.argmax(data, axis=1)

# 打印每一行中概率最大的列名
for row_index, max_column_index in enumerate(max_prob_columns):
    print(f"行 {row_index + 1}: 概率最大的列名是 {max_column_index}")

# 如果您需要将每一行的最大列名存储到一个列表中，可以使用以下代码：
max_prob_columns_list = [str(max_column_index) for max_column_index in max_prob_columns]

print("每一行概率最大的列名列表:", max_prob_columns_list)

# 创建DataFrame
df = pd.DataFrame({'最大概率列名': max_prob_columns_list})
# df.to_csv('predict_1.csv', encoding='utf-8_sig')
df.to_csv('predict_2.csv', encoding='utf-8_sig')