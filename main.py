# importing the libraries we downloaded, into the file
import tensorflow
import numpy
import matplotlib.pyplot
import seaborn # NOT a mandatory library, just allows us to visualize our datasets


# downloading and loading the MNIST dataset from tensorflow
MNIST = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = MNIST.load_data()
# x_train is the dataset of 28x28 pixel images of handwritten numbers that the model will train on
# y_train is the dataset of labels corresponding to the images in x_train
# x_test is the dataset of 28x28 pixel images of handwritten numbers that the model will test on
# y_test is the dataset of labels corresponding to the images in x_test

# When using new unfamiliar datasets, it is always a good idea to visualize the data to get a better understanding of it, we can do this using seaborn
seaborn.countplot(y_train) # Visualizes how many of each label we have in our dataset, ideally they should be evenly distributed

# When using new unfamiliar datasets, it is always a good idea to check for missing values in the dataset, or NaN values, we can do this using numpy
numpy.isnan(x_train).any()
numpy.isnan(x_test).any()
# When we test the images in this dataset, they should output False, but it may not be the case in every dataset