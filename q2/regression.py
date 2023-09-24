import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from datetime import datetime,timedelta
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import k_means
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from kmodes import kmodes


def get_train_data():
    df = pd.read_csv("data/ed_volume_time.csv",index_col=False)
    df = df.loc[:,~df.columns.str.contains('Unnamed')]

    print(df.info())

    X = df.loc[:,df.columns.str.contains('时间点')]
    Y = df.loc[:,df.columns.str.contains('volume')]

    # X.fillna(0, inplace=True)
    # Y.fillna(0, inplace=True)

    print(X)
    print(Y)

    plt.figure(figsize=(6,3))
    plt.xlabel("时间(h)")
    plt.ylabel("水肿体积(10-3ml)")
    plt.rc('font', family='Times New Roman')
    plt.subplots_adjust(bottom=0.15)

    for index in X.index[:25]:
        # plt.scatter(X.loc[index,:],Y.loc[index,:])
        plt.plot(X.loc[index,:],Y.loc[index,:],alpha=0.8)
    plt.savefig("fig/trend.jpg",dpi=600)
    plt.show()

    return X,Y

def regression(X,Y):

    models = [LinearRegression(),MLPRegressor(),RandomForestRegressor(n_estimators=100,max_depth=5)]

    testx = np.arange(0,100,1).reshape(-1,1)

    for model in models:
        # model = MLPRegressor()
        model.fit(X, Y)
        # print(model.coef_)
        # print(model.intercept_)
        print(model.score(X, Y))
        # print(model.predict(X), Y)
        print("\n\n")

        testy = model.predict(testx)
        plt.plot(testx,testy)
        plt.show()

def multi_fit(x,y):
    degree = 3
    coefficients = np.polyfit(x, y, degree)
    poly = np.poly1d(coefficients)

    # 生成拟合后的 x 和对应的 y
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = poly(x_fit)

    # 创建原始数据的散点图
    plt.scatter(x, y, label='原始数据', color='blue', marker='o')

    # 创建拟合曲线
    plt.plot(x_fit, y_fit, label=f'{degree}-次多项式拟合', color='red')
    plt.show()
    score= r2_score(y, poly(x))
    print(score)


def km():
    df_person = pd.read_csv("D:\Program Files (x86)\PyProject\math_model_revice\data\表1-患者列表及临床信息.csv")
    # 删除sub81异常数据
    df_person.drop(index=80, inplace=True,axis=0)
    df_person = df_person.loc[:160,~df_person.columns.str.contains('Unnamed')]

    data = df_person.iloc[:,3:]
    print(df_person.tail())
    print(df_person.info())

    scaler = MinMaxScaler()
    data = scaler.fit_transform(X=data)

    df = pd.read_csv("data/ed_volume_time.csv")
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    for cluster in range(3,6):

        model = KMeans(n_clusters=cluster)
        # model = kmodes.KModes(n_clusters=cluster)
        predict = model.fit_predict(data)

        label = "label_"+str(cluster)
        df[label]=predict
        df_person[label]=predict

        print(f"cluster{cluster}:")
        print(df_person.groupby(label).mean()['90天mRS'])
        print(df_person.groupby(label).count()['90天mRS'])

    df.to_csv("data/ed_km_no81.csv",encoding='utf-8_sig')


def get_all_volume(df):
    X = df.loc[:, df.columns.str.contains('时间点')]
    Y = df.loc[:, df.columns.str.contains('volume')]

    df_f = pd.DataFrame(columns=['time','volume'])

    index =0
    for i in X.index:
        for c1,c2 in zip(X.columns,Y.columns):
            if pd.isnull(X.loc[i,c1]) or pd.isnull(Y.loc[i,c2]):
                continue
            df_f.loc[index,"time"] = X.loc[i,c1]
            df_f.loc[index,"volume"] = Y.loc[i,c2]
            index+=1

    df_f.sort_values('time',inplace=True)
    df_f.reset_index(drop=True,inplace=True)

    return df_f

def gauss_function(x,a1,b1,c1,a2=0,b2=0,c2=0):
    y = a1 * np.exp(-((x-b1)/c1)**2) + a2 * np.exp(-((x-b2)/c2)**2)
    return y

def plot_box(df):
    # 设置箱子的数量和边界
    num_bins = 11  # 你可以根据需要调整箱子的数量

    # 时间等频
    # bin_edges = pd.cut(df['time'], bins=num_bins, labels=False)
    bin_edges = np.asarray([0,24,48,100,200,400,600,800,1000,2000,4000,10000])

    # 样本等频
    # bin_edges = pd.qcut(df['time'], q=num_bins, labels=False)

    # 将分箱结果添加到DataFrame
    df['分箱'] = pd.cut(df['time'], bins=bin_edges, labels=False)
    box_counts = df['分箱'].value_counts()

    # 使用Seaborn绘制箱线图
    plt.figure(figsize=(10,5))
    sns.boxplot(x='分箱', y='volume', data=df[df['分箱'].isin(box_counts.index)], palette='Set2')

    plt.xlabel("时间(h)")
    plt.ylabel("水肿大小(10-3ml)")
    plt.rc('font', family='Times New Roman')

    # 添加标签和标题
    bin_centers = bin_edges
    # bin_centers =df.groupby('分箱')['time'].min()
    # bin_centers =np.asarray([0,24,48,100,200,400,600,800,1000,2000,4000])

    custom_labels = [f'{bin_edges[i]}-{bin_edges[i + 1]}' for i in range(len(bin_edges) - 1)]
    plt.xticks(range(len(custom_labels)), custom_labels)
    # plt.xticks(range(bin_centers.shape[0]), bin_centers.astype(int))

    # 显示图形
    plt.savefig('fig/all_box.tif', dpi=600)
    plt.show()

    avg_volume = df.groupby('分箱')['volume'].median()
    avg_times = df.groupby('分箱')['time'].median()

    print("分箱时间及对应的大小输出")
    x = "x=[" + ",".join(map(str,round(avg_times,4)))+"];"
    y = "y=[" + ",".join(map(str,round(avg_volume,4))) + "];"
    print(x)
    print(y)
    print("cftool;")

    print('\n\n')


# 分聚类画图并输出箱信息
def plot_cluster5_box():
    df = pd.read_csv("data/ed_km_no81.csv")
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    cluster = 5
    label = "label_"+str(cluster)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    for i in range(cluster):

        df_tmp = df[df[label]==i]
        df_label = get_all_volume(df_tmp)

        bin_edges = np.asarray([0, 24, 48, 100, 200, 400, 600, 800, 1000, 2000, 4000, 10000])

        # 将分箱结果添加到DataFrame
        df_label['分箱'] = pd.cut(df_label['time'], bins=bin_edges, labels=False)
        box_counts = df_label['分箱'].value_counts()

        # 使用Seaborn绘制箱线图
        sns.boxplot(x='分箱', y='volume', data=df_label[df_label['分箱'].isin(box_counts.index)], palette='Set2',ax=axes[int(i/2),i%2])
        axes[int(i/2),i%2].set_ylabel('水肿大小(10-3ml)')

        # 添加标签和标题
        bin_centers = bin_edges

        custom_labels = [f'{bin_edges[i]}-{bin_edges[i + 1]}' for i in range(len(bin_edges) - 1)]

        axes[int(i/2),i%2].set_xticks(range(len(custom_labels)), custom_labels,rotation=15)

    axes[0,0].set_xlabel("")
    axes[0,1].set_xlabel("")
    axes[1,0].set_xlabel("")
    axes[1,1].set_xlabel("")
    axes[2,0].set_xlabel('时间(h)')
    fig.delaxes(axes[2, 1])
    # 显示图形
    # plt.tight_layout()
    plt.savefig('fig/cluster5_box.jpg', dpi=600)
    plt.show()



def count_err_total():
    df = pd.read_csv('data/ed_volume_time.csv')
    df = df.loc[:99,:]

    df_time = df.loc[:,df.columns.str.contains('时间点')]
    df_volume = df.loc[:,df.columns.str.contains('volume')]

    df_err = pd.DataFrame(columns=['error'])

    total_res = 0
    for i in df_time.index:
        x = df_time.loc[i,:].values
        # time_mean + vol_media
        pre_y = gauss_function(x,35780,451.6,202.9,2890,98.11,109.7)
        # pre_y = gauss_function(x,35780,451.6,202.9,2890,98.11,109.7)

        y = df_volume.loc[i,:].values

        res = 0
        count=0
        for j in range(y.shape[0]):
            if pd.isnull(y[j]):
                break

            res += abs(y[j]-pre_y[j])
            count+=1
            print(y[j],pre_y[j])
        df_err.loc[i,'error']= res/count
        print(res)
        total_res+=res
    df_err.to_csv('data/error_total.csv',encoding='utf-8_sig')
    print(total_res)

def count_err_cluster(c):
    df = pd.read_csv('data/ed_km.csv')
    df = df.loc[:99,:]

    a1=[-193800000,39120,11000,39260,20090]
    b1=[361.1,709,240.2,135.9,120.4]
    c1=[204.8,216.3,430.7,156.8,136.2]
    a2=[193900000,35930,0,33500,0]
    b2=[361.1,316.6,0,579.2,0]
    c2=[204.8,494.2,0,128.5,0]

    for cluster in range(c):
        df_c = df.loc[df[f"label_{c}"]==cluster,:]

        df_time = df_c.loc[:,df_c.columns.str.contains('时间点')]
        df_volume = df_c.loc[:,df_c.columns.str.contains('volume')]

        total_res = 0
        for i in df_time.index:
            x = df_time.loc[i,:].values
            # time_mean + vol_media
            pre_y = gauss_function(x,a1[cluster],b1[cluster],c1[cluster],a2[cluster],b2[cluster],c2[cluster])
            # pre_y = gauss_function(x,35780,451.6,202.9,2890,98.11,109.7)

            y = df_volume.loc[i,:].values

            res = 0
            count = 0
            for j in range(y.shape[0]):
                if pd.isnull(y[j]):
                    break
                res += abs(y[j]-pre_y[j])
                count+=1
                print(y[j],pre_y[j])

            print(res)
            df.loc[i,'error_cluster']=res/count
            total_res+=res
        print(total_res)
    print(df.info())
    df.to_csv(f"data/error_cluster{c}.csv",encoding='utf-8_sig')


# df = pd.read_csv("data/allVolume_no81.csv")
# plot_box(df)

for f in glob.glob("data/cluster/*.csv"):
    print(f)
    df = pd.read_csv(f)
    plt.scatter(df['time'],df['volume'])
    plt.show()