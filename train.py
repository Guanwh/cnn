# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import model  
import input_data
N_CLASSES =3   #要分类的类别数，这里是2分类，sit和stand
IMG_W = 256     #设置图片的size，resize图像，太大的话训练时间久
IMG_H = 256
BATCH_SIZE = 20
CAPACITY = 64
MAX_STEP =10000  #迭代一千次，如果机器配置好的话，建议至少10000次以上
learning_rate = 0.0001   # 学习率一般小于0.0001


#存放一些模型文件的目录
train_dir = '/home/zs/gwh/dataset/train_jpg/'
logs_train_dir = '/home/zs/gwh/dataset/log_e/'
#获取批次batch    
train,train_label= input_data.get_files(train_dir)
train_batch,train_label_batch = input_data.get_batch(train,train_label,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
#测试数据及标签  
#val_batch, val_label_batch = inputdata.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
#训练操作定义
train_logits =model.inference(train_batch,BATCH_SIZE,N_CLASSES)
train_loss = model.losses(train_logits,train_label_batch)
train_op = model.trainning(train_loss,learning_rate)
train_acc = model.evaluation(train_logits,train_label_batch)
#测试操作定义
#test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)  
#test_loss = model.losses(test_logits, val_label_batch)          
#test_acc = model.evaluation(test_logits, val_label_batch)  
#这个是log汇总记录  
summary_op = tf.summary.merge_all()
	#产生一个会话
sess = tf.Session()
	#产生一个writer来写log文件
train_writer = tf.summary.FileWriter(logs_train_dir,sess.graph)
    #产生一个saver来存储训练好的模型
saver = tf.train.Saver()
	#所有节点初始化
sess.run(tf.global_variables_initializer())
	#队列监控
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess = sess,coord = coord)
Loss=[]
Accuracy=[]
#进行batch的训练
try:
	#执行MAX_STEP步的训练，一步一个batch
    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _,tra_loss,tra_acc = sess.run([train_op,train_loss,train_acc])
	Loss.append(tra_loss)
	Accuracy.append(tra_acc)
        if step %  50 == 0:
            print('Step %d,train loss = %.2f,train occuracy = %.2f%%'%(step,tra_loss,tra_acc))
            #每迭代50次，打印出一次结果
            summary_str = sess.run(summary_op)
            train_writer.add_summary(summary_str,step)

        if step % 200 ==0 or (step +1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step = step)
            #每迭代200次，利用saver.save()保存一次模型文件，以便测试的时候使用
    print('training finished')
	#代价函数曲线
    fig1,ax1=plt.subplots(figsize=(10,7))
    plt.plot(Loss)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    plt.title('Cross Loss')
    plt.grid()
    plt.show()
    #准确率曲线
    fig7,ax7=plt.subplots(figsize=(10,7))
    plt.plot(Accuracy)
    ax7.set_xlabel('Epochs')
    ax7.set_ylabel('Accuracy Rate')
    plt.title('Train Accuracy Rate')
    plt.grid()
    plt.show()
except tf.errors.OutOfRangeError:
    print('Done training epoch limit reached')
finally:
    coord.request_stop()

