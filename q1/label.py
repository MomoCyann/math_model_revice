import pandas as pd
from datetime import datetime, timedelta

# 读取CSV文件或创建DataFrame
table1 = pd.read_csv('../data/表1-患者列表及临床信息.csv')
df = pd.read_csv('../data/表2-患者影像信息血肿及水肿的体积及位置.csv')
time_df = pd.read_csv('../data/time.csv', index_col=False)

# 假设你有一个DataFrame df，然后可以使用以下代码来选择所需的列：
selected_columns = df.columns[df.columns.str.contains('HM_volume|流水号')]

# 从DataFrame中获取选定的列
selected_df = df[selected_columns]

# 打印选定的列
print(selected_df)

# 获取所有'HM_volume'列
hm_columns = [col for col in selected_df.columns if 'HM_volume' in col]

# 将'HM_volume'列除以1000
selected_df[hm_columns] = selected_df[hm_columns].div(1000)

# 使用shift方法将第一个'HM_volume'列复制到一个新列，然后计算差值和百分比变化
for i in range(1, len(hm_columns)):
    col_name = f'diff_{hm_columns[i]}'
    selected_df[col_name] = selected_df[hm_columns[i]] - selected_df[hm_columns[0]]
    selected_df[f'percent_change_{hm_columns[i]}'] = (selected_df[hm_columns[i]] - selected_df[hm_columns[0]]) / selected_df[hm_columns[0]] * 100

# 创建一个布尔索引，检查条件1和条件2是否同时满足
condition1 = selected_df[[f'diff_{col}' for col in hm_columns[1:]]].gt(6).any(axis=1)
condition2 = selected_df[[f'percent_change_{col}' for col in hm_columns[1:]]].gt(33).any(axis=1)

# 使用逻辑运算符&将条件1和条件2组合
final_condition = condition1 | condition2

# 创建一个新列来表示是否满足条件
selected_df['hm_big'] = 0

selected_df['hm_big_time'] = ''
hm_columns_reversed = hm_columns[::-1][:-1]
for col in hm_columns_reversed:
    condition11 = selected_df[f'diff_{col}'].gt(6)
    condition22 = selected_df[f'percent_change_{col}'].gt(33)
    f_condition = condition11 | condition22
    # 创建'time'列，将condition中True的位置设为字符串'timese'
    if f_condition.any():
        column_index = selected_df.columns.get_loc(col)
        previous_column = selected_df.columns[column_index - 1]

        time_index = selected_df.loc[f_condition, previous_column]
        for index, value in time_index.items():
            # 查找它所在的行
            target_row = time_df[time_df.eq(value).any(axis=1)]
            # 如果找到目标行，获取该行的索引
            if not target_row.empty:
                row_index = target_row.index[0]
                time_column_index = time_df.columns.get_loc(previous_column)
                # 获取同一行的前一列的值
                if row_index >= 0:
                    time = time_df.iloc[row_index, time_column_index-1]
            else:
                print(f"未找到目标值{row_index} {previous_column}")
            # 计算时间差值
            dead_time = time_df.loc[row_index, '入院首次检查时间点']
            ill_time = table1.loc[row_index, '发病到首次影像检查时间间隔']
            print(timedelta(hours=48)-timedelta(hours=ill_time))
            time_difference = datetime.strptime(time, '%Y/%m/%d %H:%M') - datetime.strptime(dead_time, '%Y/%m/%d %H:%M')
            # 比较时间差值是否大于48小时
            if time_difference <= timedelta(hours=48)-timedelta(hours=ill_time):
                selected_df.loc[index, 'hm_big_time'] = str((time_difference + timedelta(hours=ill_time)).total_seconds() / 3600)
                selected_df.loc[index, 'hm_big'] = 1


# 将结果存储在新列中
result = selected_df[['hm_big', 'hm_big_time']]
result.to_csv('result3.csv', encoding='utf-8_sig')