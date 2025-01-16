# importing the libraries we downloaded, into the file
import tensorflow
import numpy
import matplotlib.pyplot as plot
import seaborn # NOT a mandatory library, just allows us to visualize our datasets


# Downloading and Loading the MNIST dataset from tensorflow
MNIST = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = MNIST.load_data()

# When using new unfamiliar datasets, it is always a good idea to visualize the data to get a better understanding of it, we can do this using seaborn
seaborn.countplot(y_train) # Visualizes how many of each label we have in our dataset, ideally they should be evenly distributed

# When using new unfamiliar datasets, it is always a good idea to check for missing values in the dataset, or NaN values, we can do this using numpy
# When we test the images in this dataset, they should output False, but it may not be the case in every dataset
numpy.isnan(x_train).any()
numpy.isnan(x_test).any()

# We know that our images are 28x28 pixels, but we need to know the shape of the images in order to train our model
# This lets the model that each input image has 28 height and width, and will have 1 color channel (Grayscale)
input_shape = (28, 28, 1) 

# We also need to reshape the data, Tensorflow expects 4D shapes, but what we are inputting above is a 3D shape
# The 4D shape is created by adding the 1 (that reprsents grayscale) to the end of the shape which was (60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train /= 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test /= 255.0

# We need to encode our labels next, as they are currently in the form of integers, but we need them to be in the form of one-hot encoded vectors
y_train = tensorflow.one_hot(y_train.astype(numpy.int32), depth=10)
y_test = tensorflow.one_hot(y_test.astype(numpy.int32), depth=10)

# We can visualize one od the images in the dataset to see what it looks like if we would like. replace "14" with the number of the image you wish to visualize
plot.imshow(x_train[14][:,:,0])
print(y_train[14])