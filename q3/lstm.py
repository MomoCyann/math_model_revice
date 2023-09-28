import pandas as pd
import numpy as np
from keras.layers import BatchNormalization, Masking, Dropout, Bidirectional, SimpleRNN
from keras.utils import to_categorical
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.metrics import AUC, F1Score, Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping
from tqdm import tqdm
from matplotlib import pyplot as plt

df_label = pd.read_csv('multiclass_label.csv', index_col=False)
# 假设你的DataFrame名为df，包含 'ID'、'Sequence' 和 'Target' 列
df = pd.read_csv('timeseries.csv', index_col=False)
df = df[~df['ID'].between('sub101', 'sub130')]


# 选择要标准化的列，从第2列（索引为1）开始
columns_to_normalize = [df.columns[2]] + df.columns[12:15].to_list() + df.columns[22:].to_list()
scaler = StandardScaler()
df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])


# 构建一个字典，其中每个ID对应其序列数据的列表
sequences_by_id = {}
for id, group in df.groupby('ID'):
    sequences_by_id[id] = group.iloc[:, 2:].values.tolist()
# 获取最长序列的长度，以便进行padding
max_seq_length = max(len(seq) for seq in sequences_by_id.values())
# 创建一个列表，用于存储每个组的序列数据，进行padding以保证相同长度
padded_sequences = []
for id, seq in sequences_by_id.items():
    # 仅保留前20列数据，后面的列用0填充
    padded_seq = pad_sequences([seq[:20]], maxlen=max_seq_length, dtype='float32', padding='post', value=-2)[0]
    padded_sequences.append(padded_seq)
# 将padded_sequences转换为NumPy数组
padded_sequences = np.array(padded_sequences)


# 训练集前100
train = padded_sequences[:100, :]
test = padded_sequences
# 获取目标值
targets = df_label['mrs'].values[:100]


def fit_a_model(X, Y, model_name, test):

    skf = RepeatedStratifiedKFold(n_repeats=5, n_splits=4, random_state=17)

    acc_scores = []
    auc_scores = []
    prec_scores = []
    rec_scores = []
    f1_scores = []
    for train_index, test_index in tqdm(skf.split(X, Y)):
        # 获取训练集与测试集
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        y_train = to_categorical(y_train, num_classes=7)
        y_test = to_categorical(y_test, num_classes=7)

        # 构建LSTM模型
        model = Sequential()
        model.add(Masking(mask_value=-2, input_shape=(train.shape[1], train.shape[2])))
        if model_name == 'LSTM':
            model.add(LSTM(128, input_shape=(train.shape[1], train.shape[2])))
        if model_name == 'BiLSTM':
            model.add(Bidirectional(LSTM(128, input_shape=(train.shape[1], train.shape[2]))))
        if model_name == 'RNN':
            model.add(SimpleRNN(128, input_shape=(train.shape[1], train.shape[2])))
        model.add(Dropout(0.2))
        model.add(Dense(7, activation='softmax'))
        # 编译模型
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                                  AUC(name='auc'),
                                                                                  Precision(name='pre'),
                                                                                  Recall(name='recall'),
                                                                                  F1Score(name='f1')])
        # 训练模型
        early_stop = EarlyStopping(monitor='val_loss', patience=3)
        model.fit(x_train, y_train, epochs=50, batch_size=16, verbose=1,
                  validation_data=(x_test, y_test), shuffle=True,
                  callbacks=[early_stop])
        predictions = model.predict(test)
        # 创建一个DataFrame来存储预测概率
        predictions_df = pd.DataFrame(predictions)
        # 保存预测概率到CSV文件
        predictions_df.to_csv('predict_2.csv', index=False, encoding='utf-8_sig')
        # evaluate the model
        scores = model.evaluate(x_test, y_test, verbose=0)

        acc_scores.append(scores[1] * 100)
        auc_scores.append(scores[2] * 100)
        prec_scores.append(scores[3] * 100)
        rec_scores.append(scores[4] * 100)
        f1_scores.append(scores[5] * 100)
    # 创建一个包含均值的字典
    data = {
        'Mean_Accuracy': [np.mean(acc_scores)],
        'Mean_AUC': [np.mean(auc_scores)],
        'Mean_Precision': [np.mean(prec_scores)],
        'Mean_Recall': [np.mean(rec_scores)],
        'Mean_F1_score': [np.mean(f1_scores)]
    }
    # 创建DataFrame
    df = pd.DataFrame(data)
    # 保存DataFrame到CSV文件
    df.to_csv(f'mean_scores_{model_name}.csv', index=False)

def draw_loss(X, Y, model_name):

    Y = to_categorical(Y, num_classes=7)
    # 构建LSTM模型
    model = Sequential()
    model.add(Masking(mask_value=-2, input_shape=(train.shape[1], train.shape[2])))
    if model_name == 'LSTM':
        model.add(LSTM(128, input_shape=(train.shape[1], train.shape[2])))
    if model_name == 'BiLSTM':
        model.add(Bidirectional(LSTM(128, input_shape=(train.shape[1], train.shape[2]))))
    if model_name == 'RNN':
        model.add(SimpleRNN(128, input_shape=(train.shape[1], train.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(7, activation='softmax'))
    # 编译模型
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 训练模型
    early_stop = EarlyStopping(monitor='val_loss', patience=3)
    history = model.fit(X, Y, epochs=50, batch_size=16, verbose=1,
                        validation_split=0.2, shuffle=True, callbacks=[early_stop])
    # 获取训练过程中的损失值和验证损失值
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # # 绘制验证集的损失曲线
    # plt.figure(figsize=(8, 6))
    # plt.plot(range(1, len(train_loss) + 1), train_loss, label='训练损失', color='deepskyblue')
    # plt.plot(range(1, len(val_loss) + 1), val_loss, label='验证损失', color='salmon')
    # plt.title(f'{model_name}模型损失变化曲线')
    # plt.xlabel('迭代次数')
    # plt.ylabel('损失')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig(f'loss_{model_name}.jpg', dpi=600)
    # plt.show()
    return train_loss, val_loss

def draw_all(train_loss_lstm, val_loss_lstm,
             train_loss_bilstm, val_loss_bilstm,
             train_loss_rnn, val_loss_rnn):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    # 创建一个图形
    plt.figure(figsize=(10, 6))
    # 绘制第一组train_loss和val_loss
    plt.plot(range(1, len(train_loss_lstm) + 1), train_loss_lstm, '-', color='orange', label='LSTM训练损失')
    plt.plot(range(1, len(train_loss_lstm) + 1), val_loss_lstm, '--', color='orange', label='LSTM验证损失')
    # 绘制第二组train_loss和val_loss
    plt.plot(range(1, len(train_loss_bilstm) + 1), train_loss_bilstm, '-', color='deepskyblue', label='BiLSTM训练损失')
    plt.plot(range(1, len(train_loss_bilstm) + 1), val_loss_bilstm, '--', color='deepskyblue', label='BiLSTM验证损失')
    # 绘制第三组train_loss和val_loss
    plt.plot(range(1, len(train_loss_rnn) + 1), train_loss_rnn, '-', color='salmon', label='RNN训练损失')
    plt.plot(range(1, len(train_loss_rnn) + 1), val_loss_rnn, '--', color='salmon', label='RNN验证损失')

    # 设置图形标题和标签
    plt.title('模型损失变化曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('Loss')
    plt.legend()

    # 显示图形
    plt.grid(True)
    plt.savefig(f'loss.jpg', dpi=600)
    plt.show()

if __name__ == '__main__':
    # fit_a_model(train, targets, 'LSTM')
    fit_a_model(train, targets, 'BiLSTM', test)
    # fit_a_model(train, targets, 'RNN')

    # train_loss_lstm, val_loss_lstm = draw_loss(train, targets, 'LSTM')
    # train_loss_bilstm, val_loss_bilstm = draw_loss(train, targets, 'BiLSTM')
    # train_loss_rnn, val_loss_rnn = draw_loss(train, targets, 'RNN')

    # draw_all(train_loss_lstm, val_loss_lstm,
    #          train_loss_bilstm, val_loss_bilstm,
    #          train_loss_rnn, val_loss_rnn)
