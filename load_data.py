import json
from collections import defaultdict
import itertools

####### Load the amazon data
# load the amazon product data
json_data = open('data/amazon_products').read()

# They didn't format the JSON properly -- this fixes that
json_data = json_data.replace('\x01', '').replace('\n', '')

# build a list of tuples where each 
# product data stream begins and ends
ix = list()
prev_ix = 0
for pos in re.finditer('}{', json_data):
    ix.append((prev_ix, pos.start() + 1))
    prev_ix = pos.start() + 1


# how many items do we want to load? 10 to test.
n = 500

# load each review as a python dict, store them in a list
amazon = [json.loads(json_data[idx[0] : idx[1]]) for idx in ix[:n]]

####### Load the hierarchy
heirarchy = open('data/AmazonHeirarchy.json').read()

labels = json.loads(heirarchy)

root_labels = {}

for lab in labels:
    root_labels[int(lab['BrowseNodeId'])] = lab['Name']

#######

def get_parents(item):
    parents = []
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
    return set(parents)




def get_top_label(item, root_labels):
    return [root_labels[it] for it in get_parents(item) if it in root_labels.keys()]



# simple example of what we can do -- we can pull 
# review content and tie it the ASIN -- the unique product ID
# We can also write out the product categories, etc.

def grab_reviews(item, reviewtype = 'PrunedEditorialReviews'):
    return [(item["ASIN"], it['Content'], get_parents()) for it in item[reviewtype]]

# This is a list of lists of product reviews.
products = [grab_reviews(t['Item']) for t in amazon]

# this just flattens
reviews = [review for item in products for review in item]


from nltk.corpus import stopwords

stop = stopwords.words('english')

def get_parents(item):
    parents = []
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
    return set(parents)



def get_top_label(item, root_labels):
    return [root_labels[it] for it in get_parents(item) if it in root_labels.keys()]


[root_labels[it] for t in amazon for it in get_parents(t['Item']) if it in root_labels.keys()]

[get_dict_of_parents(t['Item']) for t in amazon]


products = [grab_reviews(t['Item']) for t in amazon]
categories = [nodeid['BrowseNodeId'] for t in amazon for nodeid in t['Item']['BrowseNodes']['BrowseNode'] if t['Item']['BrowseNodes']['BrowseNode'].__class__ is list]



def get_labels(item, graphs):
    labs = []
    if item['BrowseNodes']['BrowseNode'].__class__ is list:
        for node in item['BrowseNodes']['BrowseNode']:
            labs += list(int(node['BrowseNodeId']))
    else:
        labs += list(int(item['BrowseNodes']['BrowseNode']['BrowseNodeId']))
    return Set([get_categories(i, graph) for i in labs])

def get_labels(item, graphs):
    labs = list(get_parents(item))
    if item['BrowseNodes']['BrowseNode'].__class__ is list:
        for node in item['BrowseNodes']['BrowseNode']:
            labs.append(int(node['BrowseNodeId']))
    else:
        labs.append(int(item['BrowseNodes']['BrowseNode']['BrowseNodeId']))
    # return labs
    return Set([cat for cat in itertools.chain.from_iterable([get_categories(i, graphs) for i in labs])])


hl_cat = [get_labels(t['Item'], graphs) for t in amazon]
hl_cat = [(i, get_labels(amazon[i]['Item'])) for i in xrange(0, n)]




