# python怎么用pandas 合并两个csv

import pandas as pd

df1 = pd.read_csv("../data/表3-水肿.csv")
# 获取第二列之后的列名
columns_to_rename = df1.columns[2:]
# 使用列名加上'_ed'来重命名列
df1.rename(columns={column: f"{column}_ed" for column in columns_to_rename}, inplace=True)

# 打印修改后的DataFrame
print(df1)

df2 = pd.read_csv("../data/表3-血肿.csv")
# 获取第二列之后的列名
columns_to_rename = df2.columns[2:]
# 使用列名加上'_ed'来重命名列
df1.rename(columns={column: f"{column}_ed" for column in columns_to_rename}, inplace=True)

#
# df = pd.merge(df1, df2, on="流水号")
# df.to_csv("test.csv", index=False)



