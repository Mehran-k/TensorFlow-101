########################################################################
# Copyright (C) 2017  Seyed Mehran Kazemi, Licensed under the GPL V3;  #
# see: <https://www.gnu.org/licenses/gpl-3.0.en.html>                  #
########################################################################

# A simple tensorflow example showing:
#   - what a computation graph is in tensorflow
#   - how to run a computation graph
#   - how a computation graph can be variablized 
#   - how a computation graph can be visualized

import tensorflow as tf

#Let's create some nodes, see the computation graph, and run it
node1 = tf.constant(1.0, name="FirstConst")
node2 = tf.constant(0.5, name="SecConst")
node3 = tf.add(node1, node2, name="AddFirstAndSec")
node4 = tf.constant(2.1, name="ThirdConst")
node5 = tf.subtract(node3, node4, name="Sub")

#The following print command will not print 1.5. It will print a tensor.
#That's because so far, we have only created a knowledge graph and node3 is just a node in the graph.
print(node3)

#To run a computation graph, we need a session.
sess = tf.Session()

#Initializing global variables.
init = tf.global_variables_initializer()
sess.run(init)

#Now we can run some nodes in our knowledge graph
print(sess.run(node5))

#When we run, say, node3, only the parts of the graph that are necessary for computing node3 are evaluated.
#That is, when we run node3, node1 is evaluated and its value is set to 1.0, node2 is evaluated and its value is set to 0.5, and then node3 is evaluated and its value is set to 1.5.
#node4 and node5 will not be evaluated when we run node3.


#All values in the above graph are constants.
#What if we want to send arguments to the graph?
#node6 defines an argument of type float in the graph.
#We will specify the value of node6 when we run the graph.
node6 = tf.placeholder('float', name="input")
node7 = tf.add(node3, node6)

#The value of the arguments is specified using feed_dict
#In the following line, we run node7 sending a value 4.0 as argument for node6.
#You can think of node7 as a function which takes in a value for node6, sums it with node3, and returns the result.
print(sess.run(node7, feed_dict={node6: 4.0}))


#We can see our computation graph using TensorBoard. The following lines specify the path where tensorflow logs and the graph are stored.
logs_path = '/tmp/tensorflow_logs/tutorial'
summary_writer = tf.summary.FileWriter(logs_path)
summary_writer.add_graph(tf.get_default_graph())

#After running this code, enter "tensorboard --logdir=/tmp/tensorflow_logs/tutorial" in the terminal.
#Then open a browser, go to "http://0.0.0.0:6006", and click on "Graphs" tab to see your computation graph.
