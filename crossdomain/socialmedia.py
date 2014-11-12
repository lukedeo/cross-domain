import sys
import json
from collections import defaultdict
import itertools
import cPickle as pickle
import re
from sets import Set

from crossdomain import get_amazon_graphs
from crossdomain.hierarchy import get_categories

import numpy as np

def get_parents(item):
    parents = []
    if item.__class__ is dict:
        try:
            parents += [item['BrowseNodeId']]
        except KeyError, e:
            parents += get_parents(item['BrowseNode'])
    elif item.__class__ is list:
        for elem in item:
            parents += get_parents(elem['Ancestors']['BrowseNode'])
    return parents

def get_labels(item, graphs):
    labs = list(get_parents(item['Amazon_Browsenodes']))
    return list(Set([cat for cat in itertools.chain.from_iterable([get_categories(i, graphs) for i in labs])]))

def grab_content(item, graphs):
    def _lambda(it):
        return {'source': it['conversation_src'], 'text' : it['conversation_text'], 'labels' : get_labels(it, graphs)}
    return (_lambda(it) for it in item)

def clean_json_socialmedia():
    """
    Creates a new JSON file properly formatted for loads
    """

    infile = open('data/Social_Conversations_AmazonLabel.json')
    outfile = open('data/Social_Conversations_AmazonLabel_clean', 'w')

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

def get_socialmedia(json_data, hierarchy, to_pickle = True):

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

    socialmedia = []
    faulty_ix = []
    ctr = 1
    total = len(ix)
    print 'Converting to JSON Dictionary...'
    for idx in ix:
        _dict = yield_json(json_data[idx[0] : idx[1]])
        if not (_dict == _ERROR_CODE):
            socialmedia.append(_dict)
        else:
            faulty_ix.append(idx)
        sys.stdout.write('{} of {} entries processed'.format(ctr, total) + ': %d%%   \r' % (100 * (float(ctr) / total)) )
        sys.stdout.flush()
        ctr += 1

    twitter = []
    ebay = []
    ctr = 1
    total = len(socialmedia)
    print '\nSplitting and Labelling...'
    for item in grab_content(socialmedia, graphs):
        if item['source'] == 'tw':
            twitter.append(item)
        else:
            ebay.append(item)
        sys.stdout.write('{} of {} entries processed'.format(ctr, total) + ': %d%%   \r' % (100 * (float(ctr) / total)) )
        sys.stdout.flush()
        ctr += 1

    if to_pickle:
        print '\nStoring...'
        with open('ebay.pkl', 'wb') as fp:
            pickle.dump(ebay, fp, pickle.HIGHEST_PROTOCOL)
        with open('twitter.pkl', 'wb') as fp:
            pickle.dump(twitter, fp, pickle.HIGHEST_PROTOCOL)

    return {'twitter' : twitter, 'ebay' : ebay}
