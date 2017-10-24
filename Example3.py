########################################################################
# Copyright (C) 2017  Seyed Mehran Kazemi, Licensed under the GPL V3;  #
# see: <https://www.gnu.org/licenses/gpl-3.0.en.html>                  #
########################################################################

# Tensorflow simple example 3 creating a logistic regression model to predict user genders from their movie ratings

#For this example, download movielens 1M dataset from https://grouplens.org/datasets/movielens/1m/
#Then run "create_matrix.py to create the pickle files"

#Movielens contains ratings of users for movies.
#In this example, we would like to predict user genders based on the movies they rated.
#For each user, the pickle file contains a vector indicating which movies the user rated (1) and which they didn't (0).

import tensorflow as tf
import numpy as np
import random
import pickle
import math

#reading the train examples (train_x), train labels (train_y), test examples (test_x) and test labels (test_x)
with open('ml1m_matrices.pickle','rb') as f:
    train_x, train_y, test_x, test_y = pickle.load(f)

num_train_users = len(train_y)
num_test_users = len(test_y)
num_items = len(train_x[0]) #items corresponds to movies in this example

x = tf.placeholder('float', [None, num_items]) #x is an argument of shape [None, num_items] having elements of type float
y = tf.placeholder('float')

#Let's create a logistic regression (LR) model
#LR has a weight for each movie, and a bias
#We initialize the weights to zero, and initialize the bias randomly from a normal distribution
weights = tf.Variable(tf.zeros((num_items, 1)))
bias = tf.Variable(tf.random_normal([1]))

#To get the predictions, we multiply the input to the weights, sum it with the bias, and take the sigmoid
prediction = tf.sigmoid(tf.matmul(x, weights) + bias)
#Sum of squares error (sse)
sse = tf.reduce_sum(tf.square(prediction - y))
#L2 regularization parameter
l2_lambda = tf.constant([0.5])
#L2 regulirizer
l2_reg = tf.reduce_sum(tf.square(weights)) * l2_lambda
#Final error
error = sse + l2_reg
#Defining an optimizer aiming at minimizing the error
optimizer = tf.train.AdamOptimizer().minimize(error)
#Calculating the accuracy
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.floor(prediction * 2), y), tf.float32))

#Creating a session and initializing it
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 1000
for i in range(1000):
	#Getting a batch of the input data
	idx = np.random.choice(np.arange(num_train_users), batch_size, replace=False)
	epoch_x, epoch_y = train_x[idx], train_y[idx]
	#Running the optimizer with the batch
	_, err = sess.run([optimizer, error], feed_dict={x: epoch_x, y:epoch_y})

	if i % 50 == 0:
		print("Iteration:", i, "Train Error:", err)

#Printing the accuracy
print("Test Accuracy:", sess.run(accuracy, feed_dict={x: test_x, y:test_y}))

