# -*- coding: utf-8 -*-
import os
import glob
from PIL import Image  
import numpy as np  
import tensorflow as tf  
import matplotlib.pyplot as plt  
import model  
import time
#from input_data import get_files  
def get_one_image(img_dir):
     image = Image.open(img_dir)
     #Image.open()
     #好像一次只能打开一张图片，不能一次打开一个文件夹，这里大家可以去搜索一下
     plt.imshow(image)
#     plt.show(image)
     image = image.resize([256,256])
     image_arr = np.array(image)
     return image_arr
def test(test_file,count0,count1,count2):
    log_dir = '/home/zs/gwh/dataset/log_e/'
    name=test_file.split('.')
    real=int(name[0][-1:])
    print('real=============',real)
    image_arr = get_one_image(test_file)
#    BATCH_SIZE = 1  
#    N_CLASSES = 2
    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,256, 256, 3])
        #print(image.shape)
        logits = model.inference(image,1,3)
        logits = tf.nn.softmax(logits)
        x = tf.placeholder(tf.float32,shape = [256,256,3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
	    sess.run(tf.global_variables_initializer())
	    print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(log_dir)
# 	    print()
	    print('ckpt=====',ckpt)
#	    print('ckpt.model_checkpoint_path=========',ckpt.model_checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                #调用saver.restore()函数，加载训练好的网络模型
                print('====================',ckpt.model_checkpoint_path.split('/'))
		print('====================',ckpt.model_checkpoint_path.split('/')[-1].split('-'))
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction) 
            print('预测的标签为：')
            print(max_index)
            print('预测的结果为：')
            print(prediction)

            if max_index==0:
                print('This is a sit with possibility %.6f' %prediction[:, 0])
                if real==0:
                    count0=count0+1
            if max_index == 1:
                print('This is a jump with possibility %.6f' %prediction[:, 1])
                if real==1:
                    count1=count1+1
	    if max_index==2:
                print('This is a lie with possibility %.6f' %prediction[:, 2])
		if real==2:
                    count2=count2+1
#	    if max_index==3:
#		print('This is a lie with possibility %.6f' %prediction[:, 3])
    return count0,count1,count2
if __name__ == '__main__':      
#    test_file = '/home/zs/gwh/dataset/test_jpg/picture/8.jpg'  
    test_dir = '/home/zs/gwh/dataset/test_jpg/picture/'
#    for i in range(1,301):
#        test_file=test_dir+str(i)+'.jpg'
    file_glob=os.path.join(test_dir,"*."+"jpg")
    file_list=[]
    file_list.extend(glob.glob(file_glob))
    l=len(file_list)
    count0=0
    count1=0
    count2=0
    time_start=time.time()
    for i in range(0,l):
        print('i===============',i)
        count0,count1,count2= test(file_list[i],count0,count1,count2)
    time_end=time.time()
    print('waste time=',time_end-time_start)
    print(count0,count1,count2)
