# Experiments to understand how the numbers of layers
# influence the classification rate of a CNN.
# ---
# by Prof. Dr.-Ing. Jürgen Brauer, www.juergenbrauer.org


import numpy as np

import tensorflow as tf
import keras
from keras.utils import to_categorical
from keras import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras import backend as K


import cv2

import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import isfile, join

import scipy.misc

from html_logger import html_logger


# Here are some settings:
FOLDER_DATA_ROOT = "data"
IMG_SIZE = (256, 256)
NUM_CLASSES = 2

# Define the input shape of the CNN:
# img_height x img_width x nr_channels
THE_INPUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1], 3)

NR_EPOCHS_TO_TRAIN = 1

LOG_FILENAME = "logfile.html"
my_logger = html_logger( LOG_FILENAME )

exp_result_dict = {}



def version_checks():

    my_logger.log_msg( "Your NumPy version is: " + np.__version__ )
    my_logger.log_msg( "Your TensorFlow version is: " + tf.__version__)
    my_logger.log_msg( "Your Keras version is: " + keras.__version__ )
    my_logger.log_msg( "Your OpenCV version is: " + cv2.__version__ )




def load_and_show_a_single_test_image():

    # 1.
    # Load image and display information about it
    filename = FOLDER_DATA_ROOT + "\\cars\\n02958343_12.JPEG"

    image = cv2.imread(filename)
    print("Type of image is", type(image))
    print("Image has dimensions :", image.shape)

    # 2.
    # OpenCV stores images in BGR color order,
    # but MatplotLib expects RGB color order
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 3.
    # Resize image to desired size
    image = cv2.resize(image, IMG_SIZE)

    # 4.
    # Show image
    plt.imshow(image)
    plt.title("One of the images")
    plt.show()



def load_images(foldername):
    filenames = \
        [f for f in listdir(foldername) if isfile(join(foldername, f))]

    # For quick testing of the rest of the code:
    # Just use the first N images
    N = 50
    filenames = filenames[0:N]

    nr_images = len(filenames)

    images = np.zeros((nr_images, IMG_SIZE[0], IMG_SIZE[1], 3))

    for img_nr, filename in enumerate(filenames):
        absolute_filename = foldername + "/" + filename
        image = cv2.imread(absolute_filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMG_SIZE)
        image = image * (1.0 / 255.0)

        # set image in 4D array
        images[img_nr] = image

    my_logger.log_msg( "I have read " +
                       str(nr_images) +
                       " images from folder "
                       + foldername )

    mb = images.nbytes / (1024.0**2)
    my_logger.log_msg( "Size of the images NumPy Array: " +
                       str(images.nbytes) +
                       " bytes = " +
                       str(mb) +
                       " MB " )
    return images



def test_permutation():

    test_array = np.arange(10).reshape((5, 2))
    permut = np.random.permutation(test_array.shape[0])
    print("permut=", permut)
    print("test_array before:", test_array)
    test_array = np.take(test_array, permut, axis=0)
    print("test_array after:", test_array)



def prepare_train_and_test_matrices(bikes_images, cars_images):

    my_logger.log_msg( "bikes_images has shape" + str(bikes_images.shape) )
    my_logger.log_msg( "cars_images has shape"  + str(cars_images.shape) )

    labels_bikes_images = np.zeros((bikes_images.shape[0], 1))
    labels_bikes_images[:, 0] = 0

    labels_cars_images = np.zeros((cars_images.shape[0], 1))
    labels_cars_images[:, 0] = 1

    X = np.vstack((bikes_images,
                   cars_images))

    my_logger.log_msg("X has shape " + str(X.shape) )

    Y = np.vstack((labels_bikes_images,
                   labels_cars_images))

    my_logger.log_msg( "Y has shape " +str(Y.shape) )

    # Shuffle data
    some_permutation = np.random.permutation(X.shape[0])
    X = np.take(X, some_permutation, axis=0)
    Y = np.take(Y, some_permutation, axis=0)

    # Split data into training and testing data
    TEST_DATA_RATIO = 0.1
    nr_train_images = int((1 - TEST_DATA_RATIO) * X.shape[0])
    my_logger.log_msg( "The " +
                       str(X.shape[0]) +
                       " many images will be split " +
                       "into train and test images." )
    X_train = X[0:nr_train_images]
    Y_train = Y[0:nr_train_images]
    X_test = X[nr_train_images:]
    Y_test = Y[nr_train_images:]

    my_logger.log_msg( "Shape of X_train is " + str(X_train.shape) )
    my_logger.log_msg( "Shape of Y_train is " + str(Y_train.shape) )

    my_logger.log_msg( "Shape of X_test is " + str(X_test.shape) )
    my_logger.log_msg( "Shape of Y_test is " + str(Y_test.shape) )

    return X_train, Y_train, X_test, Y_test



def build_a_cnn_model(nr_layers):

    model = Sequential()

    # Feature hierarchy:
    for layer_nr in range(nr_layers):

        model.add(Conv2D(32, kernel_size=(4, 4), strides=(2, 2),
                         activation='relu',
                         input_shape=THE_INPUT_SHAPE))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    # MLP part:
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    return model



def plot_curves(history, exp_name):

    my_logger.log_msg( "The following keys are stored in the training history: " +
                       str( history.history.keys() ) )
    my_logger.log_msg( "loss history has len" + str(len(history.history['loss'])) )
    my_logger.log_msg( str( history.history['loss']) )

    # Loss Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['loss'], 'r', linewidth=3.0)
    plt.plot(history.history['val_loss'], 'b', linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Loss Curves', fontsize=16)

    # log image to html logfile
    img_filename = my_logger.get_new_image_filename()
    plt.savefig(img_filename)
    my_logger.log_img_by_file(img_filename)

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)

    # log image to html logfile
    img_filename = my_logger.get_new_image_filename()
    plt.savefig( img_filename )
    my_logger.log_img_by_file( img_filename )




def test_model(model_filename, X_test, Y_test_one_hot_encoded):

    # 1. Show log message in log file
    #    that we now will test the model
    nr_test_images = X_test.shape[0]
    my_logger.log_msg("Now testing model : " + model_filename)
    my_logger.log_msg("Nr of test images : " + str(nr_test_images))
    my_logger.log_msg("Shape of X_test : " + str(X_test.shape))

    # 2. Reload the model
    model = load_model(model_filename)


    # 3. Now go through all test images
    correct_classified = 0
    for test_img_nr in range(0, nr_test_images):

        # 3.1 Get the image from the 4D NumPy array
        img_data = X_test[test_img_nr]

        # 3.2 Get ground truth label for this image
        gt_label = np.argmax(Y_test_one_hot_encoded[test_img_nr])
        my_logger.log_msg("Ground truth label is: " + str(gt_label))

        # 3.3 Let our trained model predict the class!
        img_data_as_4d_array = img_data.reshape((-1,
                                                 img_data.shape[0],
                                                 img_data.shape[1],
                                                 img_data.shape[2])
                                               )

        # 3.4 Get prediction result from neuron outputs
        neuron_outputs = model.predict(img_data_as_4d_array)
        predicted_label = np.argmax(neuron_outputs.reshape(-1))
        my_logger.log_msg("Predicted label is: " + str(predicted_label))
        my_logger.log_msg("Neuron outputs are: " + str(neuron_outputs))

        # 3.5 Convert NumPy data back to an OpenCV image
        # in order to display it correctly
        testimg = scipy.misc.toimage(img_data)

        # 3.6 Log image to html logfile
        plt.cla()
        plt.imshow(testimg)
        my_logger.log_pyplot(plt)

        # 3.7 Update statistics of correct classified examples
        if predicted_label == gt_label:
            correct_classified +=1


    # 4. Compute classification rate
    classification_rate = float(correct_classified) / float(nr_test_images)
    my_logger.log_msg( "Test results: " +
                       "Correct classified: " +
                       str(correct_classified) +
                       " of " +
                       str(nr_test_images) +
                       " --> classification rate: " +
                       str(classification_rate
                           )
                     )

    return classification_rate



def main():

    # 1. Show version numbers of important libraries
    version_checks()


    # 2. Load images of bikes and cars
    bikes_images = load_images(FOLDER_DATA_ROOT + "/bikes")
    cars_images = load_images(FOLDER_DATA_ROOT + "/cars")


    # 3. Split the data into training and test data
    X_train, Y_train, X_test, Y_test = \
        prepare_train_and_test_matrices(bikes_images, cars_images)


    # 4. Prepare one-hot-encoding of output labels
    Y_train_one_hot_encoded = to_categorical(Y_train)
    Y_test_one_hot_encoded = to_categorical(Y_test)


    # 5. Do experiments!
    experiment_nr = 0
    for EXP_PARAM_NR_LAYERS in range(1,4):

        # 5.1 Write experiment info to logfile
        experiment_nr += 1
        exp_name = str(experiment_nr).zfill(2)
        my_logger.log_msg("")
        my_logger.log_msg("-----------------------")
        my_logger.log_msg("Experiment: " + exp_name)
        my_logger.log_msg("-----------------------")
        my_logger.log_msg("Nr of conv/max-pool layers: " + str(EXP_PARAM_NR_LAYERS) )
        my_logger.log_msg("")


        # 5.2 Build a CNN model
        model = build_a_cnn_model(EXP_PARAM_NR_LAYERS)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n<br>".join(stringlist)
        my_logger.log_msg( short_model_summary )


        # 5.3 Train the model
        history = model.fit(X_train, Y_train_one_hot_encoded,
                            validation_split=0.90, batch_size=64, epochs=NR_EPOCHS_TO_TRAIN)


        # 5.4 Save the model
        model_filename = "exp_" + exp_name + ".keras_model"
        if os.path.exists(model_filename):
            os.remove(model_filename)
        my_logger.log_msg( "Saving model to file " + model_filename )
        model.save( model_filename, overwrite=True )


        # 5.5 Plot curves
        plot_curves(history, exp_name)


        # 5.6 Forget the model
        del(model)


        # 5.7 Test the model
        classification_rate = test_model(model_filename, X_test, Y_test_one_hot_encoded)


        # 5.8 Save classification rate
        exp_result_dict[experiment_nr] = classification_rate


        # 5.9 Write all results into one line in log file
        my_logger.log_msg("All experiment results so far:")
        for key, val in exp_result_dict.items():
            msg = "{} -> {}".format(key, val)
            my_logger.log_msg( msg )


    # end-for (EXP_PARAM_NR_LAYERS)


    my_logger.close()

    # due to strange error message:
    #   TypeError: 'NoneType' object is not callable
    # I used the approach of Nimi42, see:
    # see https://github.com/tensorflow/tensorflow/issues/8652
    K.clear_session()


main()

