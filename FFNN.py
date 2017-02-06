# -*- coding: utf-8 -*-

#这是最简单的前馈神经网络的代码
import tensorflow as tf
import input_data
import math

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
images_placeholder = tf.placeholder(tf.float32, shape=(None,784))
labels_placeholder = tf.placeholder("float",[None,10])


def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev = 0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1,shape = shape)
	return tf.Variable(initial)

#构建Weight_Varible 和 Bias_Variable
#这一段就是前馈神经网络的核心代码，通过和前面的卷积神经网络比较。其实他们大致的原理都是一样的。都是利用神经元来总结图像的特征
#唯一不同的就是他们总结的方式“
'''
前馈神经网络就是最简单的神经网络，说他简单就是因为他的总结方式简单。他的总结方式就是把图像从784和像素总结成200个像素
然后在从200个像素总结成50个像素，最后50个像素总结成10个像素。(这里最后一定是10个像素。因为我们最后要分成10类。所以分成几类就总结成多少个像素)注意这里有个函数叫做tf.nn.relu 这是一个激励函数，激励函数的作用
是什么呢？他的作用就是让结果非线性化，从而结果变得更加平滑，为什么要非线性化，因为曲线可以更加准确的描述点。例如一个点大多数情况下处于
一个很小的值，但是有时候会突然升高，这种情况线性函数描述不了的，必须使用非线性函数。
还有一点就是最后一步一定要用一个分类器，这样才能起到分类的效果。例如这里的tf.nn.softmax(多项式逻辑回归),那么能不能用sigmoid 这个分类函数呢？
当然不能了，因为sigmoid 只能分成俩类即0 和 1 。而我们一共有10类。

那什么是卷积神经网络。卷积神经网络已经在前面写了代码。这里主要概括一下他的总结方式
首先他有一个滤波器，你来制定滤波器的大小，高度，扫描方式，行走方式，然后用滤波器对图像进行扫描，例如我们可以指定他的大小是2×2×32让他无边框扫描，每次行走一步
然后他通过扫描会总结出一个高度为32的块，然后在用pooling对这个块进行压缩，例如我们压缩成原来的一半大小，然后在重复刚才的动作。一般情况下
都是conv俩次pooling俩次，然后我们就可以用分类器进行分类了。卷积神经网络比前馈神经网络慢的多，但是要更加准确

'''

with tf.name_scope('hidden1') as scope:
	W1 = weight_variable([784,200])
	B1 = bias_variable([200])
	hidden1 = tf.nn.relu(tf.matmul(images_placeholder, W1) + B1)
with tf.name_scope('hidden2') as scope:
	W2 = weight_variable([200,50])
	B2 = bias_variable([50])
	hidden2 = tf.nn.relu(tf.matmul(hidden1, W2) + B2)
with tf.name_scope('output') as scope:
	W3 = weight_variable([50,10])
	B3 = bias_variable([10])
	logits = tf.nn.softmax(tf.matmul(hidden2, W3) + B3)

cross_entropy = -tf.reduce_sum(labels_placeholder*tf.log(logits))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
tf.scalar_summary("loss",accuracy)
tf.scalar_summary("step",cross_entropy)
summary_op = tf.merge_all_summaries()
summary_writer = tf.train.SummaryWriter("/home/hadoop/tensorFlow_pro/ffnn_output",sess.graph)
for i in range(20000):
	batch = mnist.train.next_batch(50)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={images_placeholder:batch[0], labels_placeholder: batch[1]})
		summary_str = sess.run(summary_op,feed_dict={images_placeholder:batch[0], labels_placeholder: batch[1]})
		summary_writer.add_summary(summary_str, i)
		summary_writer.flush()
		print "i %d, training accuracy %g" %(i, train_accuracy)
	sess.run(train_step,feed_dict={images_placeholder:batch[0], labels_placeholder: batch[1]})
	
	
	#summary_str = sess.run(summary_op, feed_dict={images_placeholder:batch[0], labels_placeholder: batch[1]})
	#summary_writer.add_summary(summary_str, i)
	
    	

