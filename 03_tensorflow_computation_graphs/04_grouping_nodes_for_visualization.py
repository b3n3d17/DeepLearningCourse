# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
# Here we show how to group placeholder,
# variable, or operation nodes in the computation graph
# using the with-statement / name-scopes


import tensorflow as tf
print("Your TF version is", tf.__version__)

# 1. add input nodes to the default graph
with tf.name_scope("input-nodes") as scope:
    w1 = tf.placeholder(tf.float32, name="w1")
    w2 = tf.placeholder(tf.float32, name="w2")

print(w1)

# 2. add parameter nodes to the graph
with tf.name_scope("parameters") as scope:
    p1 = tf.Variable(1.111, name="p1")
    p2 = tf.Variable(2.222, name="p2")
    p3 = tf.Variable(3.333, name="p3")
    p4 = tf.Variable(4.444, name="p4")
    p5 = tf.Variable(5.555, name="p5")
    p6 = tf.Variable(6.666, name="p6")


# 3. add computation nodes for first hidden neuron
with tf.name_scope("neuron1") as scope:
    w3 = tf.multiply(p1,w1, name="p1w1")
    w4 = tf.multiply(p2,w2, name="p2w2")
    w5 = tf.add(w3,w4, name="w3-plus-w4")


# 4. add computation nodes for second hidden neuron
with tf.name_scope("neuron2") as scope:
    w6 = tf.multiply(p3,w1, name="p3w1")
    w7 = tf.multiply(p4,w2, name="p4w2")
    w8 = tf.add(w6,w7, name="w6-plus-w7")


# 5. add computation nodes for third neuron,
#    the output neuron
with tf.name_scope("neuron3") as scope:
    w9  = tf.multiply(p5,w5, name="p5w5")
    w10 = tf.multiply(p6,w8, name="p6w8")
    w11 = tf.add(w9,w10, name="w9-plus-w10")


# 6. add a node for initialization variables to the graph
with tf.name_scope("variable_initializer") as scope:
    init_all_vars_op = tf.global_variables_initializer()


# 7. start a computation session
s = tf.Session()


# 6. evaluate nodes
s.run(init_all_vars_op)
result = s.run(w11, feed_dict={w1:3,w2:4})


# 7. print computation result
print("f(3,4)="+str(result))


# 8. write graph to a file
writer = tf.summary.FileWriter("visu", s.graph)
writer.close()


# close session and reset default graph
s.close()
tf.reset_default_graph()





