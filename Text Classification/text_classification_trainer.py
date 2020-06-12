"""
Another large application of neural networks is text classification.
We will use a neural network to classify movie reviews as either positive or negative.
"""

import tensorflow as tf
from tensorflow import keras
import numpy

"""
The dataset we will use for these next tutorials is the IMDB movie dataset from keras.
"""
imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

"""
Having a look at our data we'll notice that our reviews are integer encoded.
This means that each word in our reviews are represented as positive integers 
where each integer represents a specific word.
This is necessary as we cannot pass strings to our neural network. However,
if we (as humans) want to be able to read our reviews and see what they look like
we'll have to find a way to turn those integer encoded reviews back into strings.
The following code will do this for us:"""

"""
We start by getting a dictionary that maps all of our words to an integer, add some more keys to it like , etc.
and then reverse that dictionary so we can use integers as keys that map to each word.
The function defied will take as a list the integer encoded reviews and return the human readable version.
"""
_word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0  # Padding
word_index["<START>"] = 1  # Start
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3  # Unused

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def decode_review(text):
    # this function will return the decoded (human readable) reviews
    return " ".join([reverse_word_index.get(i, "?") for i in text])


"""
If we have a look at some of our loaded in reviews we'll notice that they are different lengths.
This is an issue. We cannot pass different length data into out neural network.
Therefore we must make each review the same length. To do this we will follow the procedure below:
- if the review is greater than 250 words then trim off the extra words
- if the review is less than 250 words add the necessary amount of 's to make it equal to 250.

Luckily for us keras has a function that can do this for us:
"""
# The maxlen should be the max lenght item of the train_data/test_data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",
                                                        maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

"""
Finally we will define our model!
"""

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

"""
class Adam: Optimizer that implements the Adam algorithm.
https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c

class BinaryCrossentropy: Computes the cross-entropy loss between true labels and predicted labels.
https://gombru.github.io/2018/05/23/cross_entropy_loss/#:~:text=Binary%20Cross%2DEntropy%20Loss,affected%20by%20other%20component%20values.
"""
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

"""
For this specific model we will introduce a new idea of validation data.
In the last tutorial when we trained the models accuracy after each epoch on the current training data,
data the model had seen before.
This can be problematic as it is highly possible the a model can simply memorize input data
and its related output and the accuracy will affect how the model is modified as it trains.
So to avoid this issue we will sperate our training data into two sections, training and validation.
The model will use the validation data to check accuracy after learning from the training data.
This will hopefully result in us avoiding a false confidence for our model.

We can split our training data into validation data like so:
"""
x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

"""
We will train the model using the code below:
"""
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

"""
To have a look at the results of our accuracy we can do the following:
"""
results = model.evaluate(test_data, test_labels)
print(results)

test_review = test_data[0]
predict = model.predict([test_review])

print("Review: ", decode_review(test_review))
print("Prediction: ", str(predict[0]))
print("Actual:", test_labels[0])

"""
Up until this point we have simply been retraining our models every time that we wanted to use them.
This is fine for now on our small models that take only a few seconds to train
but for larger models this is not realistic.
Luckily for us keras provides a very easy way to save our models:
"""
model.save("Text Classification/model.h5")  # name it whatever you want but end with .h5
