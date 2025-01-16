# importing the libraries we downloaded, into the file
import tensorflow
import numpy
import matplotlib.pyplot as plot
import seaborn # NOT a mandatory library, just allows us to visualize our datasets

#print(tensorflow.__version__) # Checking the version of tensorflow we are using
#print(numpy.__version__) # Checking the version of numpy we are using

# Downloading and Loading the MNIST dataset from tensorflow
MNIST = tensorflow.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = MNIST.load_data()

# When using new unfamiliar datasets, it is always a good idea to visualize the data to get a better understanding of it, we can do this using seaborn
seaborn.countplot(y_train) # Visualizes how many of each label we have in our dataset, ideally they should be evenly distributed

# When using new unfamiliar datasets, it is always a good idea to check for missing values in the dataset, or NaN values, we can do this using numpy
# When we test the images in this dataset, they should output False, but it may not be the case in every dataset
print(numpy.isnan(x_train).any())
print(numpy.isnan(x_test).any())

# We know that our images are 28x28 pixels, but we need to know the shape of the images in order to train our model
# This lets the model that each input image has 28 height and width, and will have 1 color channel (Grayscale)
input_shape = (28, 28, 1) 

# We also need to reshape the data, Tensorflow expects 4D shapes, but what we are inputting above is a 3D shape
# The 4D shape is created by adding the 1 (that reprsents grayscale) to the end of the shape which was (60000, 28, 28)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train.astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test.astype('float32') / 255.0


# We need to encode our labels next, as they are currently in the form of integers, but we need them to be in the form of one-hot encoded vectors
y_train = tensorflow.one_hot(y_train.astype(numpy.int32), depth=10)
y_test = tensorflow.one_hot(y_test.astype(numpy.int32), depth=10)

# We can visualize one od the images in the dataset to see what it looks like if we would like. replace "14" with the number of the image you wish to visualize
plot.imshow(x_train[14][:,:,0])
print(y_train[14])

# Defining necessary variables for our model
batchSize = 64 # Play around with this value to see how it effects the training of the model
epochs = 5 # Play around with this value to see how it effects the training of the model
numClasses = 10 # Probably best to not play with this one, it effects our categorized outputs

#Defining our model
layer = tensorflow.keras.layers # Optinally adding a variable to access tensorflow.keras.layers in a much easier manner
model = tensorflow.keras.models.Sequential([ # Creating a sequential model, where we add layers to the model in a sequential manner(one after the other)
    
    # ReLU is an activation function which helps the network learn complex patterns in the data, it is the most widely used activation function 
    layer.Conv2D(32, (5,5), padding='same', activation='relu', input_shape=input_shape), # Our first convolutional layer, we are using 32 5x5 pixel filters to scan over the image and detect patterns. At this layer we define our input shape as well
    layer.Conv2D(32, (5,5), padding='same', activation='relu'), # Our second convolutional layer, which enforces what we learned in the first layer, with the same filters, for more complex pattern recognition
    
    layer.MaxPool2D(), # The Max Pooling layer reduces the image size by taking the maximum value of each 2x2 block, reducing our image from 28x28 to 14x14
    layer.Dropout(0.25), # This drops 25% of neurons in the layer, to prevent overfitting
    
    layer.Conv2D(64, (3,3), padding='same', activation='relu'), # Since our image is now 14x14, we can use smaller filters to detect finer details and patterns
    layer.Conv2D(64, (3,3), padding='same', activation='relu'), # padding='same' keeps the output the same size as the input
    
    layer.MaxPool2D(strides=(2,2)), # Further reduce image size, this time the 2x2 grid is explicitly defined, reducing our image to 7x7 pixels
    layer.Dropout(0.25), # Drops 25% of neurons
    layer.Flatten(), # Flattens our 2D feature maps (7x7x64) to a 1D vector
    layer.Dense(128, activation='relu'), # A fully connected layer with 128 neurons, each in charge of learning a complex combination of patterns and previous features
    layer.Dropout(0.5), # Final dropout, drops 50% of neurons at random

    # softmax is another activation function, but it is used in the final layer of a classification network, it outputs probabilities of each class
    # An example could be [0.01, 0.05, 0.02, 0.88, 0.01, 0.01, 0.00, 0.00, 0.00, 0.01], where the highest value is the predicted class (in this case the number 3 at 88%)
    layer.Dense(numClasses, activation='softmax') # Our final layer, with 10 neurons, one for each number 0-9

])

# Here is where we compile our model
model.compile(
    optimizer=tensorflow.keras.optimizers.RMSprop(epsilon=1e-08), # RMSprop is an optimizer that helps us train our model, epsilon is a small value to prevent division by zero
    loss='categorical_crossentropy', # This is the loss function, it measures how well our model is doing, and adjusts the weights of the model to minimize this value
    metrics=['acc'] # This is the metric we are using to measure the performance of our model, in this case we are using accuracy
)

# This simple function fits our training data, if our accuracy reaches 99.5% or higher, it stops training (to provent overfitting and reduce use of time and resources)
class myCallback(tensorflow.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.995):
      print("\nReached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

# This is where we train our model, we input our training data, batch size, epochs, and validation split
history = model.fit(x_train, y_train,
                    batch_size=batchSize,
                    epochs=epochs,
                    validation_split=0.1, # Represents the 10% of images in the dataset reserved for testing accuracy
                    callbacks=[callbacks]
                )

# The following code will allow us to visualize the loss and accuracy of our model through Loss & Accuracy Curves
fig, ax = plot.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color='r', label="Validation Loss")
ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['acc'], color='b', label="Training Accuracy")
ax[1].plot(history.history['val_acc'], color='r',label="Validation Accuracy")
ax[1].legend(loc='best', shadow=True)

plot.tight_layout()
plot.show()

# The following code will allow us to visualize the confusion matrix of our model, and see where it went wrong
# Predict the values from the testing dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = numpy.argmax(Y_pred,axis = 1) 
# Convert testing observations to one hot vectors
Y_true = numpy.argmax(y_test,axis = 1)
# compute the confusion matrix
confusion_mtx = tensorflow.math.confusion_matrix(Y_true, Y_pred_classes) 

plot.figure(figsize=(10, 8))
seaborn.heatmap(confusion_mtx, annot=True, fmt='g')

# Saving the Model
model.save("mnist_model.h5")
model.summary()