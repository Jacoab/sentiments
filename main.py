import nltk
import csv
import numpy as np


#
# Programed By: Jake Jongewaard
# Description: routines for loading amazon fine food reviews
#================================================================


import nltk
import numpy as np
import pandas as pd


def read_reviews(filename):
    """
    Reads a csv file of fine food reviews and loads the reviews
    into a numpy array

    :param filename:
    :return: 2-dimensional numpy array that holds each batch of reviews
    """
    review_csv = pd.read_csv(filename)
    review_array = np.array(review_csv.Text)
    trimmed_review_array = review_array[0:500000]

    review_batches = np.split(trimmed_review_array, 10)
    return review_batches


REVIEW_FILE = "Reviews.csv"

""" read_reviews(filename) test """
print("Testing read_reviews(filename) function with input of " + REVIEW_FILE)
review_batches = read_reviews(REVIEW_FILE)
print("  ", end="  ")
print(review_batches)

print('\nPrinting tokenized word: ')
print(review_batches[1][2])
print(nltk.word_tokenize(review_batches[1][2]))