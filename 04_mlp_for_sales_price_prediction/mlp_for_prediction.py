import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
import tensorflow as tf

PREDICTION_FILENAME = 'predictions_jb.csv'

# model parameters
# play with them and check whether house prediction becomes better!

# MLP topology:
NR_NEURONS_HIDDEN1 = 20   # nr of neurons in 1st hidden layer
NR_NEURONS_HIDDEN2 = 10   # nr of neurons in 2nd hidden layer
NR_NEURONS_OUTPUT  = 1

# Training parameter:
NR_TRAIN_STEPS = 100000
LEARN_RATE     = 0.001

# Which data to use to predict the house price?
#features = ['OverallQual', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'GarageCars', 'GarageArea']

# features1 --> error after 100k train steps, learn=0.001
# for a (1,20sig,10sig,1) MLP
# = 0.42
features1 = ['TotalBsmtSF']

# features2 --> error after 100k train steps, learn=0.001
# for a (2,20sig,10sig,1) MLP
# = 0.42
features2 = ['TotalBsmtSF', '1stFlrSF']

# features3 --> error after 100k train steps, learn=0.001
# for a (3,20sig,10sig,1) MLP
# = 0.31
features3 = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea']

# features4 --> error after 100k train steps, learn=0.001
# for a (4,20sig,10sig,1) MLP
# = 0.24
features4 = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'OverallQual']

# features5 --> error after 100k train steps, learn=0.001
# for a (5,20sig,10sig,1) MLP
# = 0.22
features5 = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'OverallQual', 'GarageArea']

# features6 --> error after 100k train steps
# for a (6,20sig,10sig,1)   MLP = 0.22 (learn=0.001)
# for a (6,20relu,10relu,1) MLP = 0.22 (learn=0.001)
# for a (6,20relu,10relu,1) MLP = 0.26 (learn=0.0001)
# for a (6,40relu,30relu,10relu,1) MLP = 0.34 (learn=0.0001)
# for a (6,20sig,1) MLP = 0.26 (learn=0.001)
# for a (6,20id,10id,1)   MLP = 0.26 (learn=0.001)
features6 = ['TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'OverallQual', 'GarageArea', 'GarageCars']

# set feature vector to use here!
features = features6

# Normalization factor for house sale prices
# This is important, since all the input feature values
# "live" in different intervals
# E.g. SalePrice: 50000-400000
#      TotalBsmtSF: 300-2000
#      OverallQual: 1-10
normalization_factor_per_feature = {"TotalBsmtSF": 0.001,
                                    "1stFlrSF": 0.001,
                                    "GrLivArea": 0.001,
                                    "OverallQual": 0.1,
                                    "GarageArea": 0.001,
                                    "GarageCars": 0.1,
                                    "SalePrice": 0.00001}


"""
1.
Read and analyse data:
Which features are promising to predict
the sale price of a house?
"""
def step1_read_and_analyse_data():
    # 1. read the data
    print("\n1. Reading in training and test data...")
    train_raw_data = pd.read_csv('01_kaggle_dataset_house_prices/train.csv')
    test_raw_data  = pd.read_csv('01_kaggle_dataset_house_prices/test.csv')

    # 2. now get a new data frame only with numeric values
    print("\n2. Preparing a new data frame with only "
          "numeric columns")
    numerics = ['int16', 'int32', 'int64', 'float16',
                'float32', 'float64']
    new_data_frame = train_raw_data.select_dtypes(include=numerics)

    # 3. show all numeric column names
    print("\n3. Here are only the columns which "
          "contain numeric data")
    numeric_column_names = list(new_data_frame.columns.values)
    print("There are ", len(numeric_column_names),
          "numeric columns")
    #i = 0
    #for col_name in numeric_column_names:
    #    print("numeric column #", i, ":",
    #          "column name =", col_name)
    #    i += 1

    # 4. try to find out what are columns
    #     correlated highly to sale price
    print("\n4. Here is the correlation of each "
          "numeric column with the 'SalesPrice' column:")
    saleprice_col = new_data_frame["SalePrice"]
    CORR_THRESHOLD = 0.6
    names_of_highly_correlated_cols = []
    for col_name in numeric_column_names:
        col = new_data_frame[col_name]
        pearsoncorr = pearsonr(col, saleprice_col)[0]
        #print("Pearson correlation of ", col_name,
        #      "and 'SalePrice' is", pearsoncorr)
        if (pearsoncorr > CORR_THRESHOLD) and (col_name != "SalePrice"):
            names_of_highly_correlated_cols.append(col_name)

    # 5. show names of columns that are highly correlated
    #    with 'SalePrice' column
    print("\n5. List of columns highly correlated with "
          "the sale price:")
    print("Here highly correlated means, that the Pearson "
          "correlation coefficient is above", CORR_THRESHOLD)
    print(names_of_highly_correlated_cols)

    return train_raw_data, test_raw_data

# end of function step1_read_and_analyse_data()


"""
2.
Prepare training and testing data
"""
def step2_prepare_data(train_raw_data, test_raw_data):

    # 1. for the training data, we need to know the real sale price
    train_matrix = train_raw_data["SalePrice"].values
    nr_of_train_rows = len(train_matrix)

    # 2. since train_matrix is now a 1D NumPy array with 1460 rows
    #    we reshape it to get a matrix of shape (1460,1)
    train_matrix = train_matrix.reshape(nr_of_train_rows,1)
    train_matrix = train_matrix * normalization_factor_per_feature["SalePrice"]

    # 3. for the final predictions, we need to know
    #    the Id of each house
    test_matrix = test_raw_data["Id"].values
    nr_of_test_rows = len(test_matrix)
    test_matrix = test_matrix.reshape(nr_of_test_rows,1)

    # 4. show shapes of train and test matrices
    #    to better understand what the input for the
    #    following training process is
    print("shape of train_matrix is", train_matrix.shape)
    print("shape of test_matrix is", test_matrix.shape)

    # 5. Compile columns highly correlated with the sale price
    #    into a train and test matrix
    print("\n1. Putting some columns that are highly correlated "
          "with the 'SalePrice' column in one training matrix")
    print("Columns (features) that will be used as input for the MLP:")
    print("features=", features)
    for column_name in features:

        # add column from training data
        # into training data matrix
        # and
        # for test data the test data column
        # into test data matrix

        # 5.1 get training and test column
        train_column =\
            train_raw_data[column_name].values.reshape(nr_of_train_rows,1)
        test_column  =\
            test_raw_data[column_name].values.reshape(nr_of_test_rows,1)
        #print(type(train_column))
        #print(train_column.shape)

        # 5.2 normalize the column data by
        #    the corresponding normalization factor
        train_column = train_column * normalization_factor_per_feature[column_name]
        test_column  = test_column  * normalization_factor_per_feature[column_name]

        # 5.3 add train column to train matrix
        train_matrix = np.hstack((train_matrix,train_column))
        #print(train_matrix.shape)

        # 5.4 add test column to test matrix
        test_matrix = np.hstack((test_matrix, test_column))
        #print(test_matrix.shape)

    print("Here are the first 10 rows of the train matrix")
    print(train_matrix[0:10,:])

    print("Here are the first 10 rows of the test matrix")
    print(test_matrix[0:10, :])

    # 6. some more data preprocessing is needed!
    #    since some values in the test table are nan!
    #    House #2121:
    #      - 'TotalBsmtSF' column has value "nan"
    #        probably since there is no basement?
    #    House #2577:
    #      - 'GarageArea' column has value "nan"
    #      - 'GarageCars' column has value "nan"
    #    Solution:
    #    1. count how many values are "nan" in the test matrix
    #    2. print a warning
    #    3. replace "nan" values with 0
    do_nan_preprocessing = True
    if do_nan_preprocessing:
        missing_data_items_train = np.count_nonzero(np.isnan(train_matrix))
        missing_data_items_test = np.count_nonzero(np.isnan(test_matrix))
        if missing_data_items_train > 0:
            print("WARNING! In the train matrix there are",
                  missing_data_items_train, "values which are 'nan'!")
            print("Setting all these values to 0")
        if missing_data_items_test > 0:
            print("WARNING! In the test marix there are",
                  missing_data_items_test, "values which are 'nan'!")
            print("Setting all these values to 0")

        # set "NaN" entries to 0
        where_are_NaNs_train = np.isnan(train_matrix)
        train_matrix[where_are_NaNs_train] = 0

        where_are_NaNs_test = np.isnan(test_matrix)
        test_matrix[where_are_NaNs_test] = 0


    # 7. return the training and test matrices
    #
    # train matrix has form:
    # SalePrice-House1    feature1 feature2 ... featureN
    # ...
    # SalePrice-House1460 feature1 feature2 ... featureN
    #
    # test matrix has form:
    # Id-House1    feature1 feature2 ... featureN
    # ...
    # Id-House1459 feature1 feature2 ... featureN
    #
    # Note: the number of features will depend on which features
    #       you put into the features list
    # E.g.
    # features = ['OverallQual', 'TotalBsmtSF', '1stFlrSF',
    #             'GrLivArea', 'GarageCars', 'GarageArea'] --> N=6
    # features = ['OverallQual'] --> N=1
    return train_matrix, test_matrix

# end of function step2_prepare_data()


"""
3.
Build the model
"""
def step3_build_model(nr_inputs):

    print("MLP will expect an input array of dimension 1 x nr_inputs. "
          "So 1 x", nr_inputs)

    # 1. create input placeholder
    input_node = tf.placeholder(tf.float32,
                                shape=(1, nr_inputs), name="input_node")


    # 2. create teacher node
    teacher_node = tf.placeholder(tf.float32, name="teacher_node")


    # 3. prepare 2D weight matrices & 1D bias vectors for all
    #    neuron layers in two dictionaries
    rnd_mat1 = tf.random_normal([nr_inputs, NR_NEURONS_HIDDEN1])
    rnd_mat2 = tf.random_normal([NR_NEURONS_HIDDEN1, NR_NEURONS_HIDDEN2])
    rnd_mat3 = tf.random_normal([NR_NEURONS_HIDDEN2, NR_NEURONS_OUTPUT]) # TODO!
    weights = {
        'h1':  tf.Variable(rnd_mat1),
        'h2':  tf.Variable(rnd_mat2),
        'out': tf.Variable(rnd_mat3)
    }

    biases = {
        'b1': tf.Variable(tf.random_normal(
            [NR_NEURONS_HIDDEN1])),
        'b2': tf.Variable(tf.random_normal(
            [NR_NEURONS_HIDDEN2])),
        'out': tf.Variable(tf.random_normal(
            [NR_NEURONS_OUTPUT]))
    }


    # 4. create MLP layer by layer...
    print("Creating a MLP with topology: ",
          nr_inputs, "x",
          NR_NEURONS_HIDDEN1, "x",
          NR_NEURONS_HIDDEN2, "x",
          NR_NEURONS_OUTPUT
          )

    # 4.1 hidden layer #1
    layer_1 = tf.add(tf.matmul(input_node, weights['h1']),
                     biases['b1'])
    #layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.sigmoid(layer_1)

    # 4.2 hidden layer #2
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']),
                     biases['b2'])
    #layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.sigmoid(layer_2)


    # 4.4 output layer with linear transfer function
    output_node = tf.matmul(layer_2, weights['out'])\
                  + biases['out']

    # 4.5 reshape output matrix to scalar value
    output_node = tf.reshape(output_node, [])

    # 5. create variable initializer node
    create_var_init_op = tf.global_variables_initializer()

    # 6. create loss node
    loss_node = tf.abs(teacher_node - output_node)
    optimizer_node = tf.train.GradientDescentOptimizer(LEARN_RATE).minimize(loss_node)

    # 7. return the MLP model
    #    or more exactly: graph nodes which we need to access in the following
    return [input_node,\
           teacher_node,\
           create_var_init_op,\
           loss_node,\
           optimizer_node,\
           output_node,
           weights["h1"],weights["h2"],weights["out"]
            ]

# end of function step3_build_model()


"""
Compute average error on training data
for the model learned so far
"""
def compute_avg_error(sess, model, train_matrix):

    # 1. get all nodes from the model node list
    input_node, teacher_node, var_init_node, loss_node, optimizer_node, output_node,\
        weights_h1, weights_h2, weights_out = model


    # 2. get number of training samples =
    #    nr of rows in the train matrix
    nr_train_samples = train_matrix.shape[0]

    # 3. get number of input features
    # why -1? Because the first column in the train matrix
    #         is not a feature, but the SalePrice we want to predict!
    nr_input_features = train_matrix.shape[1] - 1

    # 4. we will compute first the sum of all sample losses and
    #    then the average loss per sample
    sum_losses = 0.0
    for sample_row_nr in range(0, nr_train_samples):

        # 4.1 get the input values from all (e.g. 6) input columns
        #     exclude the first column, since this is the
        #     SalePrice!
        input_matrix = train_matrix[sample_row_nr, 1:]
        input_matrix = input_matrix.reshape(1, nr_input_features)

        # 4.2 get sales price for this house from first column
        saleprice = train_matrix[sample_row_nr, 0]
        # print("input_matrix = ", input_matrix,
        #      "--> sale price (ground truth) =", saleprice)

        # 4.3 do a single feedforward step:
        #     compute actual output y and run optimizer node
        predicted_saleprice, sample_loss =\
            sess.run([output_node, loss_node],
                     feed_dict={input_node: input_matrix,
                     teacher_node: saleprice})

        #print("predicted_saleprice =", predicted_saleprice, "sample_loss =", sample_loss)

        # 4.4 update sum of sample losses
        sum_losses += sample_loss

    # 5. compute average loss
    avg_loss = sum_losses / nr_train_samples

    # 6. return the average loss
    return avg_loss

# end of function compute_avg_error()


"""
4.
Train the model
"""
def step4_train_model(model, train_matrix, nr_steps_to_train):

    # 1. "unpack" the nodes from the model node liste
    input_node, teacher_node, var_init_node, loss_node, optimizer_node, output_node, \
        weights_h1, weights_h2, weights_out = model

    # 2. get nr of training samples
    nr_train_samples  = train_matrix.shape[0]

    # 3. get nr of input features
    # why -1? Because the first column in the train matrix
    #         is not a feature, but the SalePrice we want to predict!
    nr_input_features = train_matrix.shape[1]-1

    # 4. start a computation session
    sess = tf.Session()
    sess.run(var_init_node)

    # 5. for all training steps...
    print("I am going to do", nr_steps_to_train, "stochastic gradient descent steps!")
    print("Learn rate is", LEARN_RATE)
    for train_step in range(1, nr_steps_to_train+1):

        # 5.1 guess a random sample from the training data set
        rnd_row = np.random.randint(0, nr_train_samples)

        # 5.2 get the input values from all (e.g. 6) input columns
        input_matrix = train_matrix[rnd_row, 1:]
        input_matrix = input_matrix.reshape(1,nr_input_features)

        # 5.3 get sales price for this house
        #     we need this information to feed this value into
        #     the teacher node / placeholder
        saleprice = train_matrix[rnd_row,0]
        #print("input_matrix = ", input_matrix,
        #      "--> sale price (ground truth) =", saleprice)

        # 5.4 do a single training step:
        #    compute actual output y and run optimizer node
        actual_output, teacher_value, loss_value, _, w_h1, w_h2, w_out =\
            sess.run([output_node, teacher_node, loss_node, optimizer_node,
                      weights_h1, weights_h2, weights_out],
                      feed_dict={input_node: input_matrix,
                                 teacher_node: saleprice})

        # 5.5 from time to time, show whether we get better
        #     in predicting house prices
        if train_step % 1000 == 0:

            # compute average error on training data
            avg_error = compute_avg_error(sess, model, train_matrix)
            print("Training step ", train_step,
                  "Average error is", avg_error,
                  "actual = ", actual_output,
                  "teacher value = ", teacher_value,
                  "loss value = ", loss_value)
            # print(w_h1)

    # 6. return the session / the trained graph
    #    since we want to use the trained graph
    #    in the following to use it for predicting
    #    house prices
    return sess

#end of function step4_train_model



"""
5.
Given the trained model (with the weights of the
current session), predict the sale prices for the
houses with IDs 1461 to 2919 
See:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
"""
def step5_predict_sale_prices(sess, model, test_matrix):

    # 1.
    input_node, teacher_node, var_init_node, loss_node, optimizer_node, output_node, \
    weights_h1, weights_h2, weights_out = model

    # 2. get and show the weight matrices from variable nodes
    #    weights_h1, weights_h2, weights_out
    #w_h1, w_h2, w_out = sess.run([weights_h1, weights_h2, weights_out])
    #print("w_h1=", w_h1)
    #print("w_h2=", w_h2)
    #print("w_out=", w_out)

    # 3. get nr of test samples and nr of input features
    nr_test_samples = test_matrix.shape[0]
    nr_input_features = test_matrix.shape[1] - 1
    print("I am going to do", nr_test_samples, "predictions of house prices...")
    print("test_matrix=", test_matrix)

    # 4. now for all test samples let's predict a house price!
    prediction_matrix = np.zeros(shape=(nr_test_samples,2))
    for row_nr in range(0, nr_test_samples):

        # 4.1 get the input values from all (e.g. 6) input columns
        input_matrix = test_matrix[row_nr, 1:]
        input_matrix = input_matrix.reshape(1, nr_input_features)

        # 4.2 get the house id
        houseid = int(test_matrix[row_nr, 0])


        # 4.3 do a feedforward step:
        #    compute actual output y and run optimizer node
        predicted_saleprice = sess.run(output_node,
                                       feed_dict={input_node: input_matrix})

        # 4.4 show predicted house price
        print("House with id ", houseid,
              "has feature input_matrix = ", input_matrix,
              "--> predicted sale price is ", predicted_saleprice)

        # 4.5 save predicted house price in prediction matrix
        prediction_matrix[row_nr][0] = houseid
        prediction_matrix[row_nr][1] =\
            predicted_saleprice * (1.0/normalization_factor_per_feature["SalePrice"])


    # 5. generate a Pandas dataframe
    #    from the NumPy prediction_matrix
    predition_dataframe = pd.DataFrame({'Id'       :prediction_matrix[:,0],
                                        'SalePrice':prediction_matrix[:,1]})

    # convert column "Id" to int64 dtype
    predition_dataframe = predition_dataframe.astype({"Id": int})
    print(predition_dataframe)

    # 6. now save the Pandas dataframe to a .csv file
    predition_dataframe.to_csv(PREDICTION_FILENAME, sep=',', index=False)

# end of function step5_predict_sale_prices


def main():

    # 1. load and analyse training data
    print("\nA. Analysing data")
    train_raw_data, test_raw_data = step1_read_and_analyse_data()


    # 2. prepare training matrices
    print("\nB. Preparing training and test matrices")
    train_matrix, test_matrix =\
        step2_prepare_data( train_raw_data, test_raw_data )


    # 3. build the model
    print("\nC. Building the model")
    # why -1? Because the first column in the train matrix
    #         is not a feature, but the SalePrice we want to predict!
    nr_input_features = train_matrix.shape[1]-1
    model = step3_build_model(nr_input_features)


    # 4. train the model using the training data
    print("\nD. Training the model using the training data")
    sess = step4_train_model(model, train_matrix, NR_TRAIN_STEPS)


    # 5. let the model predict house prices
    #    for the test data and save the predictions
    print("\nE. Predicting the sale prices for the test data!")
    step5_predict_sale_prices(sess, model, test_matrix)

    # 6. close session and reset default graph
    sess.close()
    tf.reset_default_graph()


main()
