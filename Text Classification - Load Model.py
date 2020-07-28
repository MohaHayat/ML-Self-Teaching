import tensorflow as tf
from tensorflow import keras
import numpy as np

# import movie reviews data
data = keras.datasets.imdb

# split the data into testing and training data
# returns integer encoded words, each of the 10k words has an associated integer 0 - 9999
# changed from 10k o 88k
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)      # 88k most frequent words

# get mappings for words to turn integers to words we can understand
# typically you would make your own mapping with your won dict of words
# however, keras has an index for this dataset
word_index = data.get_word_index()      # returns tuples with strings

# break the tuples up into k and v where k is the key and v is the integer value
# We have 4 keys that are special chars for word mapping
# add 3 to each value so 0-3 can be used for the special characters
word_index = {k: (v+3) for k, v in word_index.items()}           # returns dict of words
word_index["<PAD>"] = 0                 # make our movie sets the same len; if 100 or 200, add padding to 100 so = 200
word_index["<START>"] = 1               #
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# swaps values and keys
# first we had word pointing to the int val but we want int val pointing to the word
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# We need to know the shape and size of our input and output
# We need to make the size and shape of data consistent -  use PAD tab
# if review is greater than 250 words, ignore
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=250)


# function that decodes training and testing data to words we can read
# returns words
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])         # try to get index i and if nothing return ?


# loading saved model from Text Classification File
model = keras.models.load_model("model.h5")

# Preprocess the sample review


def review_encode(string):
    encoded = [1]           # start with 1 because data starts at 1 (look at tags above) -- setting a starting tag
    # loop through every word and find associated integer then add to encoded list
    for word in string:
        word = word.lower()             # check for capitalization
        # check if word is in our vocab
        if word in word_index:          # word_index stores all the words corresponding to the numbers
            encoded.append(word_index[word])
        # if not, add unknown tag <UNK> ie 2
        else:
            encoded.append(2)
    return encoded


# using with you don't have to close the file afterwards
with open("Lion King Review.txt", encoding="utf-8") as f:
    # turn every word into an integer
    for line in f.readlines():
        # remove unwanted chars, use strip to remove \n, use split to split string by spaces
        nline = line.replace(",", "").replace(".", "").replace("(", "").replace(":", "").replace(")", "").replace("\"", "").strip().split(" ")
        encode = review_encode(nline)
        # Add pad and/or trim data
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding='post', maxlen=250)
        # predict model
        predict = model.predict(encode)
        print(nline)
        print(encode)
        print(predict[0])       # only doing one review at a time

