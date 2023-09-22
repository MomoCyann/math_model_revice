# b)	请以是否发生血肿扩张事件为目标变量，
# 基于“表1” 前100例患者（sub001至sub100）的个人史，疾病史，发病相关（字段E至W）、
# “表2”中其影像检查结果（字段C至X）
# 及“表3”其影像检查结果（字段C至AG，注：只可包含对应患者首次影像检查记录）等变量，
# 构建模型预测所有患者（sub001至sub160）发生血肿扩张的概率。

import pandas as pd

half_merge = pd.read_csv('half_merge.csv')
table3 = pd.read_csv('../data/表3.csv')
table3 = table3.drop('备注', axis=1)
full_merge = half_merge.join(table3.set_index("流水号"), on="流水号")


full_merge.to_csv('full_merge.csv', encoding='utf-8_sig', index=False)
