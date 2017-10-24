########################################################################
# Copyright (C) 2017  Seyed Mehran Kazemi, Licensed under the GPL V3;  #
# see: <https://www.gnu.org/licenses/gpl-3.0.en.html>                  #
########################################################################

# Tensorflow simple example 4 creating a convolutional neural network to classify MNIST images

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Input layer
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
#x is a vector, we reshape it to image format
x_image = tf.reshape(x, [-1, 28, 28, 1])

#Specifying some parameters
conv_filter_height = 5
conv_filter_length = 5
num_conv_filters_l1 = 32
num_conv_filters_l2 = 64
fully_connected_neurons = 1024

# Convolutional layer 1
W_conv1 = tf.Variable(tf.truncated_normal(shape=[conv_filter_height, conv_filter_length, 1, num_conv_filters_l1], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_conv_filters_l1]))

conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
activated_conv1 = tf.nn.relu(conv1 + b_conv1)
pooling1 = tf.nn.max_pool(activated_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Convolutional layer 2
W_conv2 = tf.Variable(tf.truncated_normal(shape=[conv_filter_height, conv_filter_length, num_conv_filters_l1, num_conv_filters_l2], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_conv_filters_l2]))

conv2 = tf.nn.conv2d(pooling1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
activated_conv2 = tf.nn.relu(conv2 + b_conv2)
pooling2 = tf.nn.max_pool(activated_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Fully connected layer 1
flat_pooling2 = tf.reshape(pooling2, shape=[-1, 7*7*64])

W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 64, fully_connected_neurons], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[fully_connected_neurons]))

activated_fc1 = tf.nn.relu(tf.matmul(flat_pooling2, W_fc1) + b_fc1)

# Dropout
keep_prob  = tf.placeholder(tf.float32)
fc1_drop = tf.nn.dropout(activated_fc1, keep_prob)

# Fully connected layer 2 (Output layer)
W_fc2 = tf.Variable(tf.truncated_normal(shape=[fully_connected_neurons, 10], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))

prediction = tf.nn.softmax(tf.matmul(fc1_drop, W_fc2) + b_fc2)

# Evaluation functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(prediction), reduction_indices=[1]))

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training algorithm
optimizer = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# Training steps
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

batch_size = 50
for i in xrange(1000):
  epoch_x, epoch_y = mnist.train.next_batch(batch_size)
  if (i % 100) == 0:
    print("Iteration:", i, "Train Accuracy", sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
  sess.run(optimizer, feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.5})

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))

