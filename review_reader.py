############################################################################
#                          ---review_reader.py--                           #
#                                                                          #
#   Provides utilities for reading and formating IMDB review files         #
#                                                                          #
############################################################################
import numpy as np
import os


neg_path = 'aclImdb_v1/aclImdb/train/neg'   # Negative reviews
pos_path = 'aclImdb_v1/aclImdb/train/pos'   # Positive reviews

# Read reviews into numpy arrays
pos_files = np.array([pos for pos in os.listdir(pos_path) if os.path.isfile(os.path.join(pos_path, pos))])
neg_files = np.array([neg for neg in os.listdir(neg_path) if os.path.isfile(os.path.join(neg_path, neg))])


def read_reviews(neg=True):
    '''
    Reads each review text file into a numpy array.

    :param neg: Review sentiment flag
    :return: Numpy array of the reviews of the specified sentiment
    '''
    reviews = []

    if neg:
        files = neg_files
        path = neg_path
    else:
        files = pos_files
        path = pos_path

    for file in files:
        with open(path + '/' + file, 'r') as reader:
            review = ''

            while True:
                char = reader.read(1)

                if not char:
                    break

                review += char

        reviews.append(review)

    return np.array(reviews)


def read_expected_word_ratings():
    '''

    :return:
    '''
    vocab_map = {}

    with open('imdb.vocab', 'r') as reader:
        vocab = [line[:len(line)-1] for line in reader]

    with open('imdbEr.txt', 'r') as reader:
        for i, line in enumerate(reader):
            vocab_map[vocab[i]] = float(line)

    return vocab_map
