from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix

# path to where images are
train_path = '/cats-and-dogs/train'
test_path = '/cats-and-dogs/test'
valid_path = '/cats-and-dogs/valid'

# build batches
# ImageDataGenerator generates batches of tensor image data
# flow from directory takes path to directory and generates batches of normalized data
# target_size size of image, classes is the classifications,
# batch_size is the batches of images we want to iterate over
train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=10)
# change batch size according to # of images you have in folder
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224, 224), classes=['dog', 'cat'], batch_size=4)

model = keras.Sequential([
    # 32 is the number of output filters - arbitrary, you can adjust over time
    # (3,3) is the kernel size - kernel size is tuple of 2 ints specifying width and height of the 2D conv window
    # which in this case is 3x3. Input shape is always in the first layer
    # Shape is based on 224 x 224 pixel images and 3 defines channel of the image, in this case RGB
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.Flatten(),             # flattens into 1D tensor
    # output layers categorizing images as 0 or 1, cat/dog
    keras.layers.Dense(2, activation='softmax')     # softmax will eval each neuron so the sum of values adds to 1
 ])

# there is a way you can add learning rate within compile but that requires more import statements
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# generator fits the model on data generated batch by batch by the Image Data Generator above
# steps per epoch is the total number of steps/batches of samples to yield from the generator before each epoch
# is finished. 40 images in total, batch size of 10, so total steps is 4. #images/batchsize = steps.
# validation step is the same calculation as above  
model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

# making predictions

test_images, test_labels = next(test_batches)               # next returns the next input line

test_labels = test_labels[:,0]                              # get first index of label. 1 or 0

# predict
# predict generator generates for input samples from our Data Generator. Passing in test batches because it is
# an Image Generator var
# steps is the total number of steps or batches of samples to yield from a generator before stopping.
# same calculation as steps per epoch
predictions = model.predict_generator(test_batches, steps=1, verbose=0)

# create confusion matrix
# [:,0] only getting first index
cm = confusion_matrix(test_labels, predictions[:,0])

# the model above results in a 50 percent accuracy. Making it no better than chance.
# the model is too simple to solve such a sophisticated problem
# we only have 3 layers within the model which clearly is not enough

# Build and fine tune VGG16 model
# Keras has a library of prebuilt models you can use and fine tune to your needs
# when you import a model it downloads to your computer
vgg16_model = keras.applications.vgg16.VGG16()

# unlike previous models, this model is not sequential. This mode is type Model
print(type(vgg16_model))

# to view model:
print(vgg16_model.summary())       # prints out summary of model: layers, output shape, and parameters

# transform this from a model type model to a model we understand, a sequential model
model = keras.Sequential()
# add each layer from vgg16 to our sequential model - the model summary will look the same
for layer in vgg16_model.layers:
    model.add(layer)

# the last layer in the model has 1000 outputs. We only want two, 0 and 1 - cat and dog
# remove the last layer
model.layers.pop()

# we want to iterate all the layers in our new model
# we freeze each layer to protect it from future training
# so that its weights will not be updated
for layer in model.layers:
    layer.trainable = False

# add a more suitable output layer with two outputs and a softmax activation
model.add(keras.layers.Dense(2, activation='softmax'))
