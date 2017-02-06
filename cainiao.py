# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#构建图表
x = tf.placeholder(tf.float32,[None,784])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

#计算损失
y_ = tf.placeholder("float",[None,10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()


#通过刚才计算的损失进行训练
#session.run(op,feed)
#执行一步tensorFlow 操作，而操作意思是运行我们定义在图中的每个operation和计算每个Tensor 的值
#This method runs one "step" of TensorFlow computation, by running the necessary graph fragment to execute every Operation and evaluate every Tensor in fetches, substituting the values in feed_dict for the corresponding input values.
with tf.Session() as sess:
	sess.run(init)
	for i in xrange(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step,feed_dict = {x:batch_xs,y_:batch_ys})
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
