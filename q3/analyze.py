import pandas as pd
from matplotlib import pyplot as plt
data = pd.read_csv('../data/表1-患者列表及临床信息.csv')


# 假设 '某一列' 是你要统计的列
value_counts_df = data['90天mRS'].value_counts().reset_index()

# 重命名列名
value_counts_df.columns = ['值', '计数']

# 绘制柱状图
plt.bar(value_counts_df['值'], value_counts_df['计数'])

# 添加标签和标题
plt.xlabel('值')
plt.ylabel('计数')
plt.title('某一列的值计数统计')

# 显示图形
plt.show()