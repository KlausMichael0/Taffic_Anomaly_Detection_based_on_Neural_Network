#!/usr/bin/env python
# coding=UTF-8  
from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

TRAIN_SPLIT = 30000

CSV_FILE_PATH = '/Users/klaus_imac/Desktop/毕设/数据集/IDS2017/Test/dataset.csv'
df = pd.read_csv(CSV_FILE_PATH)

#修改数据类型
#Object类型转换为离散数值（Label列）
df['Label'] = pd.Categorical(df['Label'])
df['Label'] = df['Label'].cat.codes
columns_counts = df.shape[1]                                                     #获取列数
for i in range(columns_counts):
  if(df.iloc[:,i].dtypes) != 'float64':
    df.iloc[:, i] = df.iloc[:,i].astype(float)

#选取11个特征和Label
features_considered = ['Bwd_Packet_Length_Min','Subflow_Fwd_Bytes','Total_Length_of_Fwd_Packets','Fwd_Packet_Length_Mean','Bwd_Packet_Length_Std','Flow_Duration','Flow_IAT_Std','Init_Win_bytes_forward','Bwd_Packets/s',
                 'PSH_Flag_Count','Average_Packet_Size']
features = df[features_considered]
data_result = df['Target']

#标准化
dataset = features.values
feature_mean = dataset.mean(axis=0)
feature_std = dataset.std(axis=0)
dataset = (dataset-feature_mean)/feature_std
dataset = pd.DataFrame(dataset,columns=features_considered)
dataset.insert(0,'Target',data_result)
dataset = dataset.values

#返回时间窗,根据给定步长对过去的观察进行采样  history_size为过去信息窗口的大小，target_size为模型需要预测的未来时间
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size                                      #如果未指定end_index,则设置最后一个训练点

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])                                      #仅仅预测未来的单个点
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


past_history = 10000
future_target = 100
STEP = 6

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 0], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)            #dataset[:,1]取最后一列的所有值
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 0],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

#训练集、验证集
BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

#创建模型
model = tf.keras.Sequential([
    layers.LSTM(32,
                input_shape=x_train_single.shape[-2:]),
    layers.Dense(32),
    layers.Dense(1, activation='sigmoid')
])

# loss = 'sparse_categorical_crossentropy'
# optimizer = tf.keras.optimizers.SGD(0.1)
model.compile(optimizer='Adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])

log_dir = "graph/log_fit/7"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train_single, y_train_single, epochs=5, batch_size=256,callbacks=[tensorboard_callback])



