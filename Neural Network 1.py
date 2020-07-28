import tensorflow as tf
from tensorflow import keras    # an API for tensorflow, essentially you write less code - high level
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist             # getting fashion_mnist data set

# like the ML series, split the data into testing and training data
# similar to how we did in first ML section
# 28 by 28 pixel images coming as arrays
(train_images, train_labels), (test_images, test_labels) = data.load_data()
# this dataset has 10 labels, each image has a specific label - 0 to 9

# label descriptions from tensorflow website
cls = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

# if you print the image, you can see the grey scale values from 0 to 256
# you can scale down the image by dividing by 255 - again scaling makes processing faster
train_images = train_images/255
test_images = test_images/255

# build the neural network:
# first layer is input later 28x28 = 784 neurons
# second layer is hidden layer where you run rectify linear unit, 128 neurons is arbitrary
# third layer is output layer where you get the probability of the image being x label decided by the neuron
# keras.Sequential builds a sequence of layers - so define in order
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),     # flatten input
    keras.layers.Dense(128, activation="relu"),  # add rectify linear unit activation function, Dense = Fully Cconnected
    keras.layers.Dense(10, activation="softmax")    # softmax will eval each neuron so the sum of values adds to 1
])

# look up the adam optimizer and sparse categorical - famous methods
# various optimizers, loss functions, and metrics
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

# train model - tweaking weights and biases
# epochs is the number of times the model sees this information - train vars
# epochs randomly pick images and labels and feed it through the NN
# the order of input images tweak the parameters of the NN
# in the end, adding epochs is supposed to increase the accuracy of the model (more is not always better)
# Also, investigate validation_set. I believe the data loaded in already splits up the data to validation and train
model.fit(train_images, train_labels, epochs=5)

# score/evaluate model
# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("Test Acc:", test_acc)

# make a prediction
# predict takes in a list or np.array
# returns a group of predictions
# we get 10 values because our output layer has 10 neurons
prediction = model.predict(test_images)
# the highest value in the list of 10 is the predicted value
# np.argmax returns index of highest value, use classification array to classify
clf = np.argmax(prediction[0])

# validate this is the correct answer
# show the input then the predicted value so we can compare
# for loop loops through some images in our test images
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)   # cmap binary gives greyscale
    plt.xlabel("Actual:" + cls[test_labels[i]])
    clf = np.argmax(prediction[i])
    plt.title("Prediction: " + cls[clf])
    plt.show()