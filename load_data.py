import json
from collections import defaultdict
import itertools
from nltk.corpus import stopwords
import cPickle as pickle
import re
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 

import numpy as np

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def get_parents(item):
    parents = []
    try:
        if item['BrowseNodes']['BrowseNode'].__class__ is list:
            for cat in item['BrowseNodes']['BrowseNode']:
                parent_categ = cat
                while parent_categ.has_key('Ancestors'):
                    parent_categ = parent_categ['Ancestors']['BrowseNode']
                parents.append(parent_categ['BrowseNodeId'])
        else:
            cat = item['BrowseNodes']['BrowseNode']
            parent_categ = cat
            while parent_categ.has_key('Ancestors'):
                parent_categ = parent_categ['Ancestors']['BrowseNode']
            parents.append(parent_categ['BrowseNodeId'])
    except Exception, e:
        parents = []
    return set(parents)

def get_labels(item, graphs):
    labs = list(get_parents(item))
    # if item['BrowseNodes']['BrowseNode'].__class__ is list:
    #     for node in item['BrowseNodes']['BrowseNode']:
    #         labs.append(int(node['BrowseNodeId']))
    # else:
    #     labs.append(int(item['BrowseNodes']['BrowseNode']['BrowseNodeId']))
    # return labs
    return list(Set([cat for cat in itertools.chain.from_iterable([get_categories(i, graphs) for i in labs])]))

def grab_reviews(item, graphs, reviewtype = 'PrunedReivews'):
    return [{'ASIN': item["ASIN"], 
             'text' : it['Content'], 
             'labels' : get_labels(item, graphs)} for it in item[reviewtype]]


def labels_to_numbers(labels, reference):
    return [reference.index(el) for el in labels]


####### Load the amazon data
# load the amazon product data
json_data = open('data/amazon_products').read()

# They didn't format the JSON properly -- this fixes that
# this 
json_data = json_data.replace('\x01', '').replace('\n', '')

# with open('amazon_products.pkl') as fp:
#     json_data = pickle.load(fp)

# build a list of tuples where each 
# product data stream begins and ends
ix = list()
prev_ix = 0
for pos in re.finditer('}{', json_data):
    ix.append((prev_ix, pos.start() + 1))
    prev_ix = pos.start() + 1


np.random.shuffle(ix)
# how many items do we want to load? 10 to test.


# load each review as a python dict, store them in a list


# simple example of what we can do -- we can pull 
# review content and tie it the ASIN -- the unique product ID
# We can also write out the product categories, etc.




n = 20000

amazon = (json.loads(json_data[idx[0] : idx[1]]) for idx in ix[:n])

reviews = [review for item in (grab_reviews(t['Item'], graphs) for t in amazon) for review in item]

corpus = (review['text'] for review in reviews)

stoplist = stopwords.words('english')

texts = ((word for word in document.lower().split() if word not in stoplist) for document in corpus)

texts = ((re.sub("\.|,|\"|-|\'|`|\*","", word) for word in text) for text in texts)

texts = ((word for word in text if (word.isalpha() & (len(word) > 1))) for text in texts)

reformatted = (' '.join(words) for words in texts)


# vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2, stop_words=stoplist)

vectorizer = CountVectorizer(min_df=2, stop_words=stoplist)

X = vectorizer.fit_transform(reformatted)


vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(corpus)


[Set(review['labels']) for review in reviews]

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform([Set(review['labels']) for review in reviews])










