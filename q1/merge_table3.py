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
df2.rename(columns={column: f"{column}_hm" for column in columns_to_rename}, inplace=True)

print(1)
df2 = df2.drop("备注", axis=1)
merged_df = df1.join(df2.set_index("流水号"), on="流水号")
merged_df.to_csv("../data/表3.csv", index=False, encoding='utf-8_sig')



