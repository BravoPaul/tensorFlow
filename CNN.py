# -*- coding: utf-8 -*-

import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##1. 构建图表
def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)

##怎么知道他是一个卷积神经网络，而不是前馈神经网络。就是因为通过他的扫描方式得到的。
##卷积神经网络大的特点就是一小块一小块的扫描样本，每个小快的大小都是一样的。
##每个神经元不断的从每个扫描的小块总结出特征和规律，从而进行预测的
def conv2d(x,W):
	#stride[1,x_movement,y_movement,1]
	#must stride[0] = 1,stride[3] = 0
	return tf.nn.conv2d(x,W,strides =[1,1,1,1],padding = 'SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'SAME')

##define placeholder for inputs to network
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])


keep_prob = tf.placeholder("float")
x_image = tf.reshape(x,[-1,28,28,1])
#print(x_image.shape) #[n_sample,28,28,1]

#conv1 layer
W_conv1 = weight_variable([5,5,1,32]) #patch 5*5,in size
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1) #output size 28*28*32
h_pool1 = max_pool_2x2(h_conv1) #output size 14*14*32

#conv2 layer
W_conv2 = weight_variable([5,5,32,64]) #patch 5*5,in size 32 out size 64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2) #output size 14*14*64
h_pool2 = max_pool_2x2(h_conv2) #output size 7*7*64

#func1 layee
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #[n_sample,7,7,64] ->> [n_sample,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)


#func2 layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)




sess = tf.InteractiveSession()

#计算loss
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess.run(tf.initialize_all_variables())
for i in range(20000):
	#取50个数据快进行训练
	batch = mnist.train.next_batch(50)
  	if i%100 == 0:
    	train_accuracy = accuracy.eval(feed_dict={
        	x:batch[0], y_: batch[1], keep_prob: 1.0})
    	print "step %d, training accuracy %g"%(i, train_accuracy)
  	#训练，通过对train_step进行训练，tensorflow会调整所有在这个session里面variable的参数
  	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print "test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})


