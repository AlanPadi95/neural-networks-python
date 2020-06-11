"""
In these first few tutorials we will simply use some built in keras datasets 
which will make loading data fairly easy. The dataset we will use to start is the Fashion MNIST datset.
This dataset contains 60000 images of different clothing/apparel items.
The goal of our network will be to look at these images and classify them appropriately
to load our first dataset in we will do the following:
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

"""
Now we will split our data into training and testing data.
It is important that we do this so we can test the accuracy of the model
on data it has not seen before.
"""
(train_images, train_labels), (test_images, test_labels) = data.load_data()

"""
Finally we will define a list of the class names and pre-process images.
We do this by dividing each image by 255. Since each image is greyscale 
we are simply scaling the pixel values down to make computations easier for our model.
"""
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

"""
Now time to create our first neural network model!
We will do this by using the Sequential object from keras.
A Sequential model simply defines a sequence of layers starting
with the input layer and ending with the output layer. Our model will have 3 layers, 
and input layer of 784 neurons (representing all of the 28x28 pixels in a picture)
a hidden layer of an arbitrary 128 neurons and an output layer of 10 neurons representing
the probability of the picture being each of the 10 classes.
"""
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),

    # Rectified linear unit activation function.
    keras.layers.Dense(128, activation="relu"),

    # Softmax converts a real vector to a vector of categorical probabilities.
    keras.layers.Dense(10, activation="softmax")
])

"""
Now that we have defined the model it is time to compile and train it.
Compiling the model is just picking the optimizer, loss function and metrics to keep track of.
Training is the process of passing our data to the model.
"""
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

"""
Now that the model has been trained it is time to test it for accuracy.
We will do this using the following line of code:
"""

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('\nTest accuracy:', test_acc)

"""
Now that we have trained the model it is time to actually use it!
We will pick a few images from our testing data, show them on the screen
and then use the model to predict what they are.

To make predictions we use our model name and .predict()
passing it a list of data to predict.
It is important that we understand it is used to make MULTIPLE predictions
and that whatever data it is expecting must be inside of a list.
Since it is making multiple predictions it will also return to use a list of predicted values.
"""
predictions = model.predict(test_images)

"""
Now we will display the first 5 images and their predictions using matplotlib.
"""
plt.figure(figsize=(5, 5))
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])
    plt.title(class_names[np.argmax(predictions[i])])
    plt.show()
