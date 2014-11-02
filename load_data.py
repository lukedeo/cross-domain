import json
from collections import defaultdict

# load the amazon product data
json_data = open('data/amazon_products').read()

# They didn't format the JSON properly -- this fixes that
json_data = json_data.replace('\x01', '').replace('\n', '')

heirarchy = open('data/AmazonHeirarchy.json').read()

labels = json.loads(heirarchy)

root_labels = {}

for lab in labels:
    root_labels[int(lab['BrowseNodeId'])] = lab['Name']


# build a list of tuples where each 
# product data stream begins and ends
ix = list()
prev_ix = 0
for pos in re.finditer('}{', json_data):
    ix.append((prev_ix, pos.start() + 1))
    prev_ix = pos.start() + 1

# how many items do we want to load? 10 to test.
n = 100

# load each review as a python dict, store them in a list
amazon = [json.loads(json_data[idx[0] : idx[1]]) for idx in ix[:n]]


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

