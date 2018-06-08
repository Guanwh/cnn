# -*- coding: utf-8 -*-
import os
import numpy as np
from PIL import Image
import tensorflow as tf
import math
import matplotlib.pyplot as plt
#导入必要的包
#-------------------------------生成图片路径和标签的list——————————————————————————————————

#存放用来训练的图片的路径
train_dir = '/home/zs/gwh/dataset/train_jpg/'
#humans =[{},{},....50000]
#human = {head:[12,23],neck:[26,61],}

#定义存放各类别数据和对应标签的列表，列表名对应你所需要分类的列别名
#sit，stand是我的数据集中要分类图片的名字
sit = []
label_sit = []
#stand = []
#label_stand = []
jump = []
label_jump = []
lie = []
label_lie = []
#step1：获取'/home/zs/gwh/dataset/train_jpg/'下所有的图片路径名，存放到  
#对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir): 
    for filename in os.listdir(file_dir):
	print('filename============',filename)
        for pic in os.listdir(file_dir+filename):
		name = pic.split('.')
		if name[0][-3:]=='sit':   
			sit.append(file_dir+filename+"/"+pic)
			label_sit.append(0)  
	        #根据图片的名称，对图片进行提取，这里用.来进行划分
                #注意，多分类问题一定要将分类的标签从0开始。
		#这里是三类，标签为0，1，2。	
#		if name[0][-5:]=='stand':
#			stand.append(file_dir+filename+"/"+pic)
#			label_stand.append(1)
		if name[0][-4:]=='jump':
                        jump.append(file_dir+filename+"/"+pic)
			label_jump.append(1)		
		if name[0][-3:]=='lie':
			lie.append(file_dir+filename+"/"+pic)
			label_lie.append(2)
	   #打印出提取图片的情况，检测是否正确提取
    print('There are %d sit\nThere are %d jump\nThere are %d lie\n' \
          %(len(sit),len(jump),len(lie)))

#    print('There are %d stand\n' \
#          %(len(stand)))  
#step2对生成的图片路径和标签list做打乱处理，把sit和stand合起来组成一个list用来水平合并数组  
    image_list = np.hstack((sit,jump,lie))
    label_list = np.hstack((label_sit,label_jump,label_lie))
#利用shuffle打乱顺序	
    temp = np.array([image_list,label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
	
#从打乱的temp中再取出list（img和lab），并返回两个list	
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    label_list = [int(i) for i in label_list]
    #将所有的img和lab转换成list  
#    all_image_list = list(temp[:, 0])  
#    all_label_list = list(temp[:, 1])  
    #print('image_list=====',image_list)
    #print('label_list=====',label_list)
    return  image_list,label_list
#    return tra_images, tra_labels, val_images, val_labels
 #--------------------------------生成Batch----------------------------------------------   
#step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab  
#是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像  
#image_W, image_H, ：设置好固定的图像高度和宽度  
#设置batch_size：每个batch要放多少张图片  
#capacity：一个队列最大多少
def get_batch(image,label,image_W,image_H,batch_size,capacity):
    #tf.cast()用来做类型转换,转化为tensorflow数据格式
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int32)
    
	#加入队列,实现一个输入的队列。
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
#    print('input_queue========',input_queue)
 #   print('label==============',label)	
	#read img from a queue
    image_contents = tf.read_file(input_queue[0])
#step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等	
	#jpeg或者jpg格式都用decode_jpeg函数，其他格式可以去查看官方文档
    image = tf.image.decode_jpeg(image_contents,channels=3)
    
#step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    #resize
    #对resize后的图片进行标准化处理
    image = tf.image.per_image_standardization(image)
	
#step4：生成batch  
#image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32   
#label_batch: 1D tensor [batch_size], dtype=tf.int32     

    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads=16,capacity = capacity)
#重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
#    print('label_batch=============',label_batch)
#    print('image_batch=============',image_batch)
    return image_batch,label_batch
    #获取两个batch，两个batch即为传入神经网络的数据
