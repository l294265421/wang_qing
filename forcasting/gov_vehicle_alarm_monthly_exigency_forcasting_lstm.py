import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from preprocess.gov_vehicle_alarm_monthly_preprocess_exigency_for_lstm import exigency_data
from data.raw_data import data_dir

save_mode_dir = 'model/exigency/'
predict_file_path = data_dir + 'exigency_data_predict.csv'
train = False

#生成训练集
#设置常量
time_step=6      #时间步
rnn_unit=100       #hidden layer units
batch_size=60     #每一批次训练多少个样例
input_size=1      #输入层维度
output_size=1     #输出层维度
lr=0.0001         #学习率
epoch = 10

# 导入数据
data_train = []
valid_data = []
for one_sample in exigency_data:
    # 留后两个月预测
    input_size = len(one_sample[0])
    data_train.append(one_sample[:-2])
    valid_data.append(one_sample[-2:])

train_x,train_y=[],[]   #训练集
for one_sample in data_train:
    for i in range(len(one_sample) - time_step - 1):
        x = one_sample[i:i + time_step]
        y = [[month[0]] for month in one_sample[i + 1:i + time_step + 1]]
        train_x.append(x)
        train_y.append(y)

train_x_array = np.array(train_x)
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)
train_x = np.nan_to_num(((train_x_array - mean) / std)).tolist()

#——————————————————定义神经网络变量——————————————————
X=tf.placeholder(tf.float32, [None,time_step,input_size])    #每批次输入网络的tensor
Y=tf.placeholder(tf.float32, [None,time_step,output_size])   #每批次tensor对应的标签
#输入层、输出层权重、偏置
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
        }

#——————————————————定义神经网络变量——————————————————
def lstm(batch):      #参数：输入网络批次数目
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    # cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    cell = tf.nn.rnn_cell.GRUCell(rnn_unit)
    init_state=cell.zero_state(batch,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states

# 训练
def train_lstm():
    global batch_size
    pred,_=lstm(batch_size)
    #损失函数
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # epoch
        for i in range(epoch):
            step=0
            start=0
            end=start+batch_size
            while(end<len(train_x)):
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[start:end],Y:train_y[start:end]})
                start+=batch_size
                end=start+batch_size
                #每10步保存一次参数
                if step%10==0:
                    print(i,step,loss_)
                    print("保存模型：",saver.save(sess, save_mode_dir + 'lstm.model'))
                step+=1

# 预测
def prediction():
    pred,_=lstm(1)      #预测时只输入[1,time_step,input_size]的测试数据
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint(save_mode_dir)
        saver.restore(sess, module_file)

        predicts = []
        sample_index = 0
        for sample in data_train:
            prev_seq = np.array(sample)
            prev_seq = prev_seq[-1 * time_step:]
            predict = []
            # 得到之后2个预测结果
            for i in range(3):
                next_seq = sess.run(pred, feed_dict={X: [prev_seq]})
                predict += next_seq[-1].tolist()
                # 每次得到最后一个时间步的预测结果，与之前的数据加在一起，形成新的测试样本
                if i < len(valid_data[0]):
                    prev_seq = np.vstack((prev_seq[1:], valid_data[sample_index][i]))
            predict = [str(num) for num in predict]
            predicts.append(predict)
            sample_index += 1
        with open(predict_file_path, mode='w', encoding='utf-8') as predict_file:
            for predict in predicts:
                predict_file.write(' '.join(predict) + '\n')

if train:
    train_lstm()
else:
    prediction()