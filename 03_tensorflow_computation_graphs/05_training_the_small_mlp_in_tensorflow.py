# TensorFlow introduction
# by Prof. Dr. Jürgen Brauer, www.juergenbrauer.org
# ---
#
# Now, we show how to train the small 3 neuron MLP /
# computation graph

import tensorflow as tf
print("Your TF version is", tf.__version__)

import numpy as np

# Parameters for training data generation
gt_p1 = 1.111
gt_p2 = 2.222
gt_p3 = 3.333
gt_p4 = 4.444
gt_p5 = 5.555
gt_p6 = 6.666

# Training parameters
NR_TRAINING_STEPS = 500
LEARN_RATE = 0.01


def generate_data(NrSamples):

    data = []
    for i in range(0, NrSamples):

        # 1.1 guess a random w1,w2 coordinates
        w1 = np.random.rand()
        w2 = np.random.rand()

        # 1.2 compute corresponding output value y

        # examples of y-values corresponding to our model
        #y = gt_p5*(w1*gt_p1 + w2*gt_p2) + gt_p6*(w1*gt_p3+w2*gt_p4)

        # example of y-values of other model!
        y = 2*w1+3*w2

        # 1.3 store that training sample
        sample = (w1, w2, y)
        data.append(sample)

    return data



def build_graph():

    # 1. add input nodes to the default graph
    w1 = tf.placeholder(tf.float32, name="w1")
    w2 = tf.placeholder(tf.float32, name="w2")

    # 2. add parameter nodes to the graph
    p1 = tf.Variable(np.random.rand(), name="p1")
    p2 = tf.Variable(np.random.rand(), name="p2")
    p3 = tf.Variable(np.random.rand(), name="p3")
    p4 = tf.Variable(np.random.rand(), name="p4")
    p5 = tf.Variable(np.random.rand(), name="p5")
    p6 = tf.Variable(np.random.rand(), name="p6")

    # 3. add computation nodes for first hidden neuron
    w3 = tf.multiply(p1, w1, name="p1w1")
    w4 = tf.multiply(p2, w2, name="p2w2")
    w5 = tf.add(w3, w4, name="w3-plus-w4")

    # 4. add computation nodes for second hidden neuron
    w6 = tf.multiply(p3, w1, name="p3w1")
    w7 = tf.multiply(p4, w2, name="p4w2")
    w8 = tf.add(w6, w7, name="w6-plus-w7")

    # 5. add computation nodes for third neuron,
    #    the output neuron
    w9 = tf.multiply(p5, w5, name="p5w5")
    w10 = tf.multiply(p6, w8, name="p6w8")
    w11 = tf.add(w9, w10, name="w9-plus-w10")

    # 6. add a node for initialization variables to the graph
    init_all_vars_op = tf.global_variables_initializer()

    return w1,w2, w11, init_all_vars_op, [p1,p2,p3,p4,p5,p6]



def compute_avg_error(sess, w1, w2, out,
                      teachervalue, loss_computer_node, test_data):

    sum_losses = 0.0
    NrTestSamples = len(test_data)

    for i in range(0, NrTestSamples):

        # 1. get next test sample
        test_sample = test_data[i]
        w1_val = test_sample[0]
        w2_val = test_sample[1]
        t_val  = test_sample[2]

        # 2. forward step
        y, current_loss = sess.run([out, loss_computer_node],
                                   feed_dict={w1:w1_val, w2:w2_val, teachervalue:t_val})

        # 3. compute sum of all samples losses
        sum_losses += current_loss

    # compute average error
    avg_error = sum_losses / NrTestSamples

    return avg_error



def train_model(w1,w2, out, teachervalue, loss_computer_node, optimizer_node,
                varinitop, train_data, test_data, params):

    # 1. start a computation session
    sess = tf.Session()

    # 2. initialize all the variables
    sess.run(varinitop)

    # 2. train the model in a loop
    NrTrainingSamples = len(train_data)
    for stepnr in range(0, NR_TRAINING_STEPS):

        # 2.1 guess a random sample index
        rnd_idx = np.random.randint(0, NrTrainingSamples)

        # 2.2 get corresponding training triple
        train_sample = train_data[rnd_idx]
        w1_val = train_sample[0]
        w2_val = train_sample[1]
        t_val = train_sample[2]

        # 2.3 compute actual output y and run optimizer node
        y, _ = sess.run([out,optimizer_node],
                         feed_dict={w1:w1_val, w2:w2_val, teachervalue:t_val})

        # 2.4 compute average error on test data
        avg_error = compute_avg_error(sess, w1, w2, out,
                                      teachervalue, loss_computer_node, test_data)

        # 2.5 show info about current learn step
        #     and current loss
        print("After training step " + str(stepnr) +
              " --> avg_error = {0:.5f} ".format(avg_error))


    # write graph to a file
    writer = tf.summary.FileWriter("visu", sess.graph)
    writer.close()

    return sess




def test_model(sess, w1, w2, out, nr_tests, test_data):

    print("Testing the model...")

    NrTestSamples = len(test_data)

    for test_nr in range(0, nr_tests):

        # get a test sample
        rnd_idx = np.random.randint(0, NrTestSamples)
        train_sample = test_data[rnd_idx]
        w1_val = train_sample[0]
        w2_val = train_sample[1]
        t = train_sample[2]

        # compute actual output y and
        # compute gradient of error function
        y = sess.run(out, feed_dict={w1: w1_val, w2: w2_val})

        # show actual output and desired output
        print("w1=",w1_val, "w2=",w2_val, " --> "
              "actual output = "+str(y)+" vs. should = "+str(t))



def main():

    # 1. generate train and test data
    train_data = generate_data(1000)
    test_data = generate_data(100)

    # 2. build the model
    w1, w2, out, varinitop, params = build_graph()

    # 3. prepare a loss computation node and an optimizer node
    teachervalue = tf.placeholder(tf.float32, name="teacher")
    loss_computer_node = (teachervalue-out)*(teachervalue-out)
    optimizer_node = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss_computer_node)

    # 4. train the model
    sess = train_model(w1, w2, out, teachervalue, loss_computer_node, optimizer_node,
                       varinitop, train_data, test_data, params)

    # 5. test the model
    test_model(sess, w1, w2, out, 20, test_data)

    # 6. close session and reset default graph
    sess.close()
    tf.reset_default_graph()


main()

