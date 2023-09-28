import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
data = pd.read_csv('../data/表1-患者列表及临床信息.csv')


# 假设 '某一列' 是你要统计的列
value_counts_df = data['90天mRS'].value_counts().reset_index()

# 重命名列名
value_counts_df.columns = ['值', '计数']
value_counts_df.to_csv('count.csv', encoding='utf-8_sig')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 绘制柱状图
plt.bar(value_counts_df['值'], value_counts_df['计数'], color=sns.color_palette("pastel"))
# 设置Y轴刻度为整数
plt.yticks(range(0, max(value_counts_df['计数']) + 1, 5))  # 以5为间隔设置刻度
# 添加标签和标题
plt.xlabel('mRS')
plt.ylabel('计数')
plt.title('sub001-sub100患者的mRS分布情况')
plt.savefig('count.jpg', dpi=600)
# 显示图形
plt.show()