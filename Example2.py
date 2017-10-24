########################################################################
# Copyright (C) 2017  Seyed Mehran Kazemi, Licensed under the GPL V3;  #
# see: <https://www.gnu.org/licenses/gpl-3.0.en.html>                  #
########################################################################

# Tensorflow simple example 2 showing:
#  - the concept of variables in tensorflow
#  - how variables can be defined
#  - how an objective function (or error function) can be defined
#  - how to update variables to maximize the objective function (or minimize error function)

import tensorflow as tf

#We take two values as input arguments
x = tf.placeholder('float', name="Arg1")
y = tf.placeholder('float', name="Arg2")

#Variables are trainable parameters
#Suppose we would like to find the average of the two input arguments.
average = tf.Variable(0.0)
#Let's define the error as the sum of the square of the distances
error = tf.square(average - x) + tf.square(average - y)

#Now let's define an optimizer which aims at minimizing the error
#0.1 is the learning rate
optimizer = tf.train.AdamOptimizer(0.1).minimize(error)

#Create the session and initialize it
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#Let's run the optimizer 500 times (500 training iterations)
for _ in range(500):
	sess.run(optimizer, feed_dict={x: 1.1, y: 1.9})

print(sess.run(average))
#This will print 1.5 which is the average of 1.1 and 1.9.
#Running the optimizer takes the derivatives of the error function w.r.t. the variables automatically and updates the variables to minimize the error.
