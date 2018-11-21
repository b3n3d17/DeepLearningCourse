# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
# Here we model the complete small "3 neuron MLP"
# 2 inputs --> 2 hidden neurons --> 1 output neuron


import tensorflow as tf
print("Your TF version is", tf.__version__)

# 1. add input nodes to the default graph
w1 = tf.placeholder(tf.float32, name="w1")
w2 = tf.placeholder(tf.float32, name="w2")
print(w1)
print(w2)


# 2. add parameter nodes to the graph
p1 = tf.Variable(1.111, name="p1")
p2 = tf.Variable(2.222, name="p2")
p3 = tf.Variable(3.333, name="p3")
p4 = tf.Variable(4.444, name="p4")
p5 = tf.Variable(5.555, name="p5")
p6 = tf.Variable(6.666, name="p6")


# 3. add computation nodes
w3 = tf.multiply(p1,w1, name="p1w1")
w4 = tf.multiply(p2,w2, name="p2w2")
w5 = tf.add(w3,w4, name="w3_plus_w4")

w6 = tf.multiply(p3,w1, name="p3w1")
w7 = tf.multiply(p4,w2, name="p4w2")
w8 = tf.add(w6,w7, name="w6_plus_w7")

w9  = tf.multiply(p5,w5, name="p5w5")
w10 = tf.multiply(p6,w8, name="p6w8")
w11 = tf.add(w9,w10, name="w9_plus_w10")


# 4. add a node to the graph
# #  for initialization of variables
init_all_vars_op = tf.global_variables_initializer()


# 5. start a computation session
s = tf.Session()


# 6. evaluate nodes
s.run(init_all_vars_op)
result = s.run(w11, feed_dict={w1:3,w2:4})


# 7. print computation result
print("f(3,4)="+str(result))


# close session and reset default graph
s.close()
tf.reset_default_graph()





