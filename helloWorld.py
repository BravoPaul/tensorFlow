import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')
sess = tf.Session()
print sess.run(hello)
a = tf.constant(10)
b = tf.constant(32)
print sess.run(a+b)
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1,input2)

with tf.Session() as sess:
	print sess.run([output],feed_dict = {input1:[7.],input2:[2.]})