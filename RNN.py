# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_input = 28
n_step = 28
n_hidden_unis = 128
n_class = 10

#tf Graph input
x = tf.placeholder(tf.float32,[None,n_step,n_input])
y = tf.placeholder(tf.float32,[None,n_class])

#Define weight
weights = {
	#(28,128)
	'in':tf.Variable(tf.random_normal([n_input,n_hidden_unis])),
	#(128,10)
	'out':tf.Variable(tf.random_normal([n_hidden_unis,n_class]))
}

biases = {
	#(128,)
	'in':tf.Variable(tf.constant(0.1,shape = [n_hidden_unis,])),
	'out':tf.Variable(tf.constant(0.1,shape = [n_class,]))
}

def RNN(X,weights,biases):
	#hidden layer for input to cell 
	#X(128 batch,28 steps, 28 input)
	#X==>(128*28,28 input)
	X = tf.reshape(X,[-1,n_input])
	#X_in==>(128batch * 28steps,128 hidden)
	X_in = tf.matmul(X,weights['in'])+biases['in']
	#X_in==>(128batch , 28steps,128 hidden)
	X_in = tf.reshape(X_in,[-1,n_step,n_hidden_unis])
	
	#cell
	###############################
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis,forget_bias=1.0)
	#lstm_cell is divided into two parts (c_stage,m_state)
	_init_state = lstm_cell.zero_state(batch_size,dtype = tf.float32)

	#the result of operation
	'''
	time major 用来确定是否我们的step 在input 的主要维度，如果是就是true 如果不是就是false
	'''
	'''
	TensorFlow 提供了多种方法用来创建一个循环神经网络（reccurent neutal network），而tf.nn.dynamic_rnn 只是其中的一种最常用的方法.这里官方文档用的是constructing 这个词，即，他其实不是创造，他是将已经存在的神经元以RNN的方式连接起来，组成一个RNN。所以你如果想构建必须要有神经元，而这个神经元是通过tf.nn.rnn_cell.BasicLSTMCell(num_units, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tanh) 来创建的
	注意这里num_units的意义是：在一个神经元中用多少维的向量来模拟输入的每一行。并不是有多少个神经元的意思。这个函数就是负责创建一个神经元，而用多少个神经元组成这个神经网络是tf.nn.dynamic_rnn 的事情
	'''
	outputs,state = tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state = _init_state,time_major = False)

	#hidden layer for output as the final state
	#################################################
	#result = tf.matmul(state[1],weights['out'])+biases['out']
	'''
	也可以这么写:
	因为在这里最后一个output就等于state[1] . 也就是我们主线程
	'''
	outputs = tf.unpack(tf.transpose(outputs,[1,0,2]))
	'''
	这里我们取得就是最后一个output，因为我们是把图片一行一行的输入到rnn的，然后用w[‘in’] 代表输入的Gate，即一个图片中用来模拟每行数据的w，
	w[‘out’]代表输出的gate，即我最后输出的w，这里的输出指的是输出到分类器的w。并不是指输出到下一个神经元的。
	而我们所作的训练其实就是训练这俩个gate，在tf.nn.rnn_cell.BasicLSTMCell（） 函数中用forget_bias来表示是否去遗忘主线程（即前一个神经元对他的输入）。
	所以我们图片的每一行out是没用。我们想要的就是最后一个output.所以rnn的本质是对一个图片，通过对w[in]来训练得知我这一行需要怎么分析，用那些数据。 w[out]是我这一行结果在
	最后的结果一种什么样的方式呈现。 
	添加俩个gate的原因是如果不添加很容易在梯度下降的时候出现极大或者极小这种极端情况
	'''
	result = tf.matmul(outputs[-1],weights['out'])+biases['out']
	

	return result

pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred,y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
	sess.run(init)
	step = 0
	while step*batch_size<training_iters:
		batch_xs,batch_ys = mnist.train.next_batch(batch_size)
		batch_xs = batch_xs.reshape([batch_size,n_step,n_input])
		sess.run([train_op],feed_dict = {
			x:batch_xs,
			y:batch_ys
		})
		if step%20==0:
			print(sess.run(accuracy,feed_dict = {
				x:batch_xs,
				y:batch_ys
			}))
		step = step+1