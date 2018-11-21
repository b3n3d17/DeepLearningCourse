# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
# Shows how to construct a MLP with an arbitrary number
# of neurons in each layer using the idea of
# matrix multiplication

import tensorflow as tf
import numpy as np

# 1. set network & training parameters
N_INPUT  = 2
N_HIDDEN1 = 10
N_HIDDEN2 = 5
N_OUTPUT = 2

N_TRAIN_STEPS = 1000000
LEARN_RATE = 0.01


# 2. prepare input placeholders
x = tf.placeholder("float")
t = tf.placeholder("float")


# 3. prepare weight matrices
W1 = tf.Variable(tf.random_normal([N_INPUT,   N_HIDDEN1]))
W2 = tf.Variable(tf.random_normal([N_HIDDEN1, N_HIDDEN2]))
W3 = tf.Variable(tf.random_normal([N_HIDDEN2, N_OUTPUT]))


# 4. create the model:
#    A 3-layer MLP (without bias inputs)
act1 = tf.matmul(x,W1)
out1 = tf.nn.sigmoid(act1)
act2 = tf.matmul(out1,W2)
out2 = tf.nn.sigmoid(act2)
y = tf.matmul(out2,W3)


# 5. create node for computing the loss
loss = tf.reduce_mean(tf.squared_difference(t, y))
optimizer =\
    tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss)


# 6. start session, initialize all variables
varinitop = tf.global_variables_initializer()
s = tf.Session()
s.run(varinitop)


# 7. training
for i in range(0,N_TRAIN_STEPS):

    rnd1 = np.random.rand()
    rnd2 = np.random.rand()
    t1 = rnd1+2.3456*rnd2
    t2 = np.sin(rnd2)
    input_vec = np.array([[rnd1, rnd2]])
    teacher_vec = np.array([[t1,t2]])
    yvalues,_ = s.run([y,optimizer], feed_dict={x: input_vec, t: teacher_vec})
    if (i % 10000 == 0):
        print("i=", i,
              "t1={0:.3f}".format(t1), "vs. y1={0:.3f}".format(yvalues[0][0]),
              "t2={0:.3f}".format(t2), "vs. y2={0:.3f}".format(yvalues[0][1]))

s.close()
tf.reset_default_graph()


