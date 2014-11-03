import json
from collections import defaultdict
import itertools
from nltk.corpus import stopwords

from gensim import corpora, models, similarities
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

def get_top_label(item, root_labels):
    return [root_labels[it] for it in get_parents(item) if it in root_labels.keys()]


# simple example of what we can do -- we can pull 
# review content and tie it the ASIN -- the unique product ID
# We can also write out the product categories, etc.

def grab_reviews(item, reviewtype = 'PrunedReviews'):
    return [{'ASIN': item["ASIN"], 'text' : it['Content'], 'labels' : get_labels(item, graphs)} for it in item[reviewtype]]

reviews = [review for item in (grab_reviews(t['Item']) for t in amazon) for review in item]

corpus = [review['text'] for review in reviews]

stoplist = stopwords.words('english')

texts = [[word for word in document.lower().split() if word not in stoplist] for document in corpus]

texts = [[re.sub("\.|,|\"|-|\'|`|\*","", word) for word in text] for text in texts]

texts = [[word for word in text if (word.isalpha() & (len(word) > 1))] for text in texts]

all_tokens = sum(texts, [])

tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) == 1)

texts = [[word for word in text if word not in tokens_once] for text in texts]

reformatted = [' '.join(words) for words in texts]





from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=2, stop_words=stoplist)

X = vectorizer.fit_transform(reformatted)


vectorizer = TfidfVectorizer(min_df=1)
vectorizer.fit_transform(corpus)












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


[root_labels[it] for t in amazon for it in get_parents(t['Item']) if it in root_labels.keys()]

[get_dict_of_parents(t['Item']) for t in amazon]


products = [grab_reviews(t['Item']) for t in amazon]
categories = [nodeid['BrowseNodeId'] for t in amazon for nodeid in t['Item']['BrowseNodes']['BrowseNode'] if t['Item']['BrowseNodes']['BrowseNode'].__class__ is list]



def get_labels(item, graphs):
    labs = list(get_parents(item))
    if item['BrowseNodes']['BrowseNode'].__class__ is list:
        for node in item['BrowseNodes']['BrowseNode']:
            labs.append(int(node['BrowseNodeId']))
    else:
        labs.append(int(item['BrowseNodes']['BrowseNode']['BrowseNodeId']))
    # return labs
    return list(Set([cat for cat in itertools.chain.from_iterable([get_categories(i, graphs) for i in labs])]))


hl_cat = [get_labels(t['Item'], graphs) for t in amazon]
hl_cat = [(i, get_labels(amazon[i]['Item'])) for i in xrange(0, n)]




