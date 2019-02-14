import tensorflow as tf
import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from data_process import data_feature_engineering
from evaluate_model import evaluate_model_1
from sklearn.model_selection import train_test_split
import tensorflow.contrib.keras as kr
seq_length = 145  # 序列长度
num_classes = 6  # 类别数
num_filters = 256  # 卷积核数目
kernel_size = 5  # 卷积核尺寸
hidden_dim = 128  # 全连接层神经元
dropout_keep_prob = 0.5  # dropout保留比例
learning_rate = 1e-3  # 学习率
vocab_size = 145  # 词汇表达小
embedding_dim = 145  # 词向量维度


data_train = pd.read_csv("D:\Competition\Happiness\data\happiness_train_abbr_new.csv")
train_np = data_feature_engineering(data_train)#处理数据，将数据转换成0，1格式和归一化数据
print("train data process end")
y = train_np[:,0]#将第一列的happiness数据取出来当作标签
#y.to_csv("D:\Competition\Happiness\data\y.csv",index =False)
x = train_np[:, 1:]#将后面的数据取出来当作输入
x_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.1)#将数据分割成训练集和测试集
y_train = kr.utils.to_categorical(Y_train, num_classes=6)

tf.reset_default_graph()
X_holder = tf.placeholder(tf.float32, [None, seq_length])
Y_holder = tf.placeholder(tf.float32, [None, num_classes])

conv = tf.layers.conv1d(X_holder, num_filters, kernel_size)
max_pooling = tf.reduce_max(conv, reduction_indices=[1])
full_connect = tf.layers.dense(max_pooling, hidden_dim)
full_connect_dropout = tf.contrib.layers.dropout(full_connect, keep_prob=1)
full_connect_activate = tf.nn.relu(full_connect_dropout)
softmax_before = tf.layers.dense(full_connect_activate, num_classes)
predict_Y = tf.nn.softmax(softmax_before)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_holder, logits=softmax_before)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)

isCorrect = tf.equal(tf.argmax(Y_holder, 1), tf.argmax(predict_Y, 1))
accuracy = tf.reduce_mean(tf.cast(isCorrect, tf.float32))

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print('cnn initing is end')
import random
for i in range(3000):
    selected_index = random.sample(list(range(len(y_train))), k=64)
    batch_X = x_train[selected_index]
    batch_Y = y_train[selected_index]
    session.run(train, {X_holder:batch_X, Y_holder:batch_Y})
    step = i + 1
    if step % 100 == 0:
        selected_index = random.sample(list(range(len(y_train))), k=200)
        batch_X = x_train[selected_index]
        batch_Y = y_train[selected_index]
        loss_value, accuracy_value = session.run([loss, accuracy], {X_holder:batch_X, Y_holder:batch_Y})
        print('step:%d loss:%.4f accuracy:%.4f' %(step, loss_value, accuracy_value))

