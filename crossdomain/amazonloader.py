import sys
import json
import itertools
import cPickle as pickle
import re
from sets import Set


# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# from nltk import word_tokenize          
# from nltk.stem import WordNetLemmatizer 
# from nltk.corpus import stopwords

from crossdomain.hierarchy import get_amazon_graphs
from crossdomain.hierarchy import get_categories

import numpy as np

# class LemmaTokenizer(object):
#     def __init__(self):
#         self.wnl = WordNetLemmatizer()
#     def __call__(self, doc):
#         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

def get_parents(item, node_access_key = 'BrowseNodes'):
    parents = []
    try:
        if item[node_access_key]['BrowseNode'].__class__ is list:
            for cat in item[node_access_key]['BrowseNode']:
                parent_categ = cat
                while parent_categ.has_key('Ancestors'):
                    parent_categ = parent_categ['Ancestors']['BrowseNode']
                parents.append(parent_categ['BrowseNodeId'])
        else:
            cat = item[node_access_key]['BrowseNode']
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

    replacements = {'\x01' : '', 
                    '\n' : ''}

    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)
    infile.close()
    outfile.close()

####### Load the amazon data
# load the amazon product data


# with open('amazon_products.pkl') as fp:
#     json_data = pickle.load(fp)

# build a list of tuples where each 
# product data stream begins and ends
# ix = list()
# prev_ix = 0
# for pos in re.finditer('}{', json_data):
#     ix.append((prev_ix, pos.start() + 1))
#     prev_ix = pos.start() + 1


# np.random.shuffle(ix)
# how many items do we want to load? 10 to test.


# load each review as a python dict, store them in a list


# simple example of what we can do -- we can pull 
# review content and tie it the ASIN -- the unique product ID
# We can also write out the product categories, etc.

_ERROR_CODE = 'JSON-ERROR'

def yield_json(string):
    try:
        return json.loads(string) 
    except Exception, e:
        return _ERROR_CODE

def product_partition(json_data, hierarchy, n_partitions = 10, prefix = 'amazon-partition'):

    # json_data = open(filename).read()

    print 'Loading product hierarchy...' 
    
    graphs = hierarchy

    print 'Indexing product...'
    ix = list()
    prev_ix = 0
    for pos in re.finditer('}{', json_data):
        ix.append((prev_ix, pos.start() + 1))
        prev_ix = pos.start() + 1


    np.random.shuffle(ix)

    amazon = []
    faulty_ix = []


    partition = []

    start = 0
    jump = len(ix) / n_partitions
    for x in xrange(0, n_partitions):
        end = min(start + jump, len(ix))
        partition.append((start, end))
        start = end

    which_partition = 0

    for start, end in partition:
        print 'Working on partition {} of {}'.format(which_partition + 1, len(partition))
        print '-------------------------------'
        indices = ix[start:end]

        amazon = []
        faulty_ix = []
        ctr = 1

        print 'Converting to JSON Dictionary...'
        for idx in indices:
            _dict = yield_json(json_data[idx[0] : idx[1]])
            if not (_dict == _ERROR_CODE):
                amazon.append(_dict)
            else:
                faulty_ix.append(idx)

            sys.stdout.write('{} of {} entries processed'.format(ctr, len(indices)) + ': %d%%   \r' % (100 * (float(ctr) / len(indices))) )
            sys.stdout.flush()
            ctr += 1

        reviews = []
        ctr, total = 1, len(amazon)
        print '\nLabeling...'
        for item in (grab_reviews(t['Item'], graphs) for t in amazon if not (t == _ERROR_CODE)):
            if not item == []:
                for review in item:
                    reviews.append(review)
            sys.stdout.write('{} of {} entries processed'.format(ctr, total) + ': %d%%   \r' % (100 * (float(ctr) / total)) )
            sys.stdout.flush()
            ctr += 1
        which_partition += 1
        print '\nStoring...'
        with open('{}-{}-of-{}.pkl'.format(prefix, which_partition, len(partition)), 'wb') as fp:
            pickle.dump((amazon, reviews), fp, pickle.HIGHEST_PROTOCOL)




if __name__ == '__main__':
    print 'Loading Data...'
    json_data = open('data/amazon_products_cleaned').read()
    print 'Parsing hierarchy...'
    graphs = get_amazon_graphs()

    n_partitions = sys.argv[1]
    product_partition(json_data, graphs, n_partitions = 1000)




# reviews_ = [review for item in (grab_reviews(t['Item'], graphs) for t in amazon) for review in item]

# # amazon = (json.loads(json_data[idx[0] : idx[1]]) for idx in ix[n:200000])

# # reviews_ = [review for item in (grab_reviews(t['Item'], graphs) for t in amazon) for review in item]


# stoplist = stopwords.words('english')

# corpus = (review['text'] for review in reviews)
# texts = ((word for word in document.lower().split() if word not in stoplist) for document in corpus)

# texts = ((re.sub("\.|,|\"|-|\'|`|\*","", word) for word in text) for text in texts)

# texts = ((word for word in text if (word.isalpha() & (len(word) > 1))) for text in texts)

# reformatted = (' '.join(words) for words in texts)


# # vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2, stop_words=stoplist)

# vectorizer = CountVectorizer(min_df=2, stop_words=stoplist)

# X = vectorizer.fit_transform(reformatted)


# vectorizer = TfidfVectorizer(min_df=1)
# X = vectorizer.fit_transform(corpus)


# [Set(review['labels']) for review in reviews]

# mlb = MultiLabelBinarizer()
# Y = mlb.fit_transform([Set(review['labels']) for review in reviews])










