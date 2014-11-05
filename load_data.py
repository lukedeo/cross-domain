import sys
import json
from collections import defaultdict
import itertools
import cPickle as pickle
import re

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

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

def clean_json_products():
    """
    Creates a new JSON file properly formatted for loads
    """

    infile = open('data/amazon_products')
    outfile = open('data/amazon_products_cleaned', 'w')

    replacements = {'\x01':'\n'}

    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)
    infile.close()
    outfile.close()

####### Load the amazon data
# load the amazon product data
json_data = open('data/amazon_products_cleaned').read()

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


_ERROR_CODE = 'JSON-ERROR'


n = 20000

def yield_json(string):
    try:
        return json.loads(string) 
    except Exception, e:
        return _ERROR_CODE

amazon = []
faulty_ix = []

# indices = ix[n:]
indices = ix
ctr = 1
for idx in indices:
    _dict = yield_json(json_data[idx[0] : idx[1]])
    if _dict is not _ERROR_CODE:
        amazon.append(_dict)
    else:
        faulty_ix.append(idx)

    sys.stdout.write('{} of {} entries processed'.format(ctr, len(indices)) + ': %d%%   \r' % (100 * (float(ctr) / len(indices))) )
    sys.stdout.flush()
    ctr += 1


reviews = []
ctr, total = 1, len(amazon)

for item in (grab_reviews(t['Item'], graphs) for t in amazon if not (t == _ERROR_CODE)):
    if not item == []:
        for review in item:
            reviews.append(review)
    sys.stdout.write('{} of {} entries processed'.format(ctr, total) + ': %d%%   \r' % (100 * (float(ctr) / total)) )
    sys.stdout.flush()
    ctr += 1




reviews_ = [review for item in (grab_reviews(t['Item'], graphs) for t in amazon) for review in item]

# amazon = (json.loads(json_data[idx[0] : idx[1]]) for idx in ix[n:200000])

# reviews_ = [review for item in (grab_reviews(t['Item'], graphs) for t in amazon) for review in item]


stoplist = stopwords.words('english')


['bow_']

corpus = (review['text'] for review in reviews)
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










