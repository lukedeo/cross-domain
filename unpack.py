import cPickle as pickle
import json
from collections import defaultdict
import itertools
import cPickle as pickle
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np


with open('subset/feature-dump.pkl', 'rb') as fp:
	X = pickle.load(fp)
with open('subset/feature-extractor-dump.pkl', 'rb') as fp:
	vectorizer = pickle.load(fp)
with open('subset/label-dump.pkl', 'rb') as fp:
	Y = pickle.load(fp)
with open('subset/label-generator-dump.pkl', 'rb') as fp:
	multilabelbinarizer = pickle.load(fp)


# now, X is a numpy matrix with Bag of Words for each row
# vectorizer is the thing that can convert new text to the bag of words
# Y is the label matrix
# multilabelbinarizer can convert labels to binary, and back





