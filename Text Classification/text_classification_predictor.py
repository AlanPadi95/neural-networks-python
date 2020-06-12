import tensorflow as tf
from tensorflow import keras
import numpy
"""
Now that we have saved a trained model we never need to retrain it!
We can simply load a saved model in by using the following.
Simply ensure that the .h5 file is in the same directory as your python script.
"""
model = keras.models.load_model("model.h5")

imdb = keras.datasets.imdb

_word_index = imdb.get_word_index()

word_index = {k: (v + 3) for k, v in _word_index.items()}
word_index["<PAD>"] = 0  # Padding
word_index["<START>"] = 1  # Start
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3  # Unused

"""
Now it is time to used our saved model to make predictions.
This is a little bit harder than it looks because we need to consider the following:
– our model accepts integer encoded data
– our model needs reviews that are of length 250 words

This means we can’t just pass any string of text into our model.
It will need to be reshaped and reformed to meet the criteria above.

Transforming our Data
The data I’ll use for this tutorial will be simple raw text data of a movie review
of one of my favorite movies, the lion king.
I’m storing this data in a text file called “the_lion_king.txt”.

To start we will need to integer encode the data.
We will do this using the following function:
"""
def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


"""
Next we will open our text file,
read in each of the reviews (in this case just one) and use the model to predict whether it is positive or negative.
"""
with open("the_lion_king.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "")\
            .replace(".", "")\
            .replace("(", "")\
            .replace(")", "")\
            .replace(":", "")\
            .replace("\"", "")\
            .strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)  # make the data 250 words long
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])
