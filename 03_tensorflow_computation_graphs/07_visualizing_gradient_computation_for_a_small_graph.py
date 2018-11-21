# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
import tensorflow as tf
print("Your TF version is", tf.__version__)

# 1. add input nodes to the default graph
x1 = tf.placeholder(tf.float32, name="x1")
x2 = tf.placeholder(tf.float32, name="x2")


# 2. add parameter nodes to the graph
p1 = tf.Variable(2.0, name="p1")
p2 = tf.Variable(3.0, name="p2")


# 3. add computation nodes
w3 = tf.multiply(p1,x1, name="p1x1")
w4 = tf.add(p2,w3, name="p2w3")


# 4. add a node for initialization variables to the graph
init_all_vars_op = tf.global_variables_initializer()


teachervalue = tf.placeholder(tf.float32, name="teacher")
loss_computer_node = (teachervalue-w4)**2
LEARN_RATE = 0.001
optimizer_node = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss_computer_node)


# 5. start a computation session
s = tf.Session()


# 6. evaluate nodes
s.run(init_all_vars_op)
result = s.run(w4, feed_dict={x1:2,x2:3})


# 7. print computation result
print("f(2,3)="+str(result))


# 8. write graph to a file
writer = tf.summary.FileWriter('visu', s.graph)
writer.close()


# close session and reset default graph
s.close()
tf.reset_default_graph()