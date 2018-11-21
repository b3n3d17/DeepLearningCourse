# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
# Here we model a simple first neuron
# with two inputs (w1,w2) and two weights (p1,p2)
# and an identity transfer function
# and want to understand how we "run" the graph

import tensorflow as tf
print("Your TF version is", tf.__version__)

# 1. add input nodes to the default graph
w1 = tf.placeholder(tf.float32, name="w1")
w2 = tf.placeholder(tf.float32, name="w2")
print(w1)
print(w2)


# 2. add parameter nodes to the graph
p1 = tf.Variable(2.0, name="p1")
p2 = tf.Variable(3.0, name="p2")
print(p1)
print(p2)


# 3. add computation nodes
w3 = tf.multiply(p1,w1, name="p1w1")
w4 = tf.multiply(p2,w2, name="p2w2")
w5 = tf.add(w3,w4, name="w3_plus_w4")
print(w3)
print(w4)
print(w5)


# 1st try:
#s = tf.Session()
#s.run(w5)

# 2nd try:
#s = tf.Session()
#s.run(w5, feed_dict={w1:10,w2:20})

# 3rd try:
#init_all_vars_op = tf.global_variables_initializer()
#s = tf.Session()
#s.run(w5, feed_dict={w1:10,w2:20})

# 4th try:
init_all_vars_op = tf.global_variables_initializer()
s = tf.Session()
s.run(init_all_vars_op)
result = s.run(w5, feed_dict={w1:10, w2: 20})
print("result="+str(result))


# close session and reset default graph
s.close()
tf.reset_default_graph()





