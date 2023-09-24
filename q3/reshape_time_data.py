import pandas as pd

df2 = pd.read_csv('../data/表2-患者影像信息血肿及水肿的体积及位置.csv', index_col=False)
df1 = pd.read_csv('multiclass_label_time.csv', index_col=False)
df3 = pd.read_csv('../data/表3.csv', index_col=False).iloc[:, 1:]
# 遍历表2的每一行
for index, row in df2.iterrows():
    person_id = row['ID']

    # 从表1中提取相同id的数据
    matching_rows = df1[df1['ID'] == person_id]

    # 寻找包含 '流水号' 的列名，确定第N次检查的数据
    if person_id == 'sub132':
        print(2)
    n = 0  # 初始化检查次数为0
    for col_name in df2.columns:
        if '流水号' in col_name:
            if pd.notnull(df2.loc[index, col_name]):
                n += 1
                if n > 1:  # 忽略第一次检查的列
                    col_index = df2.columns.get_loc(col_name)  # 获取包含 '流水号' 的列的索引
                    check_data = row[col_index:(col_index + 23)].tolist()  # 提取第N次检查的数据  # 提取第N次检查的数据
                    # 如果在表1中找到匹配的行，将数据新增在该行下面
                    if not matching_rows.empty:
                        for index, matching_row in matching_rows.iterrows():
                            # 创建一个新行，将person_id和第N次检查的数据合并
                            # 根据check_data的第一个数据在表3中查询列名为该数据的行
                            code = check_data[0]
                            if code == 20170107000727.0:
                                print('1')
                                print('2017')
                            try:
                                df3_row = df3[df3['流水号'] == code].iloc[:, 1:].values.tolist()[0]
                            except IndexError:
                                print(person_id, col_name)
                            df1_row = df1[df1['ID'] == person_id].iloc[:, 2:22].values.tolist()[0]
                            new_row = [person_id] + [check_data[0]] + df1_row + check_data[1:] + df3_row
                            # 将新行插入到匹配行的下方
                            df1 = df1.append(pd.Series(new_row, index=df1.columns), ignore_index=True)
            else:
                break
df1 = df1.groupby('ID').apply(lambda group: group.sort_values(by='流水号')).reset_index(drop=True)
print('done')
# 使用ffill方法将空值替换为上一格的值
df1.fillna(method='ffill', inplace=True)
df1.to_csv('timeseries.csv', encoding='utf-8_sig', index=0)
