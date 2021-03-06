#!/usr/bin/env python
from crossdomain.socialmedia import get_socialmedia

import crossdomain as domain
from crossdomain.hierarchy import get_amazon_graphs
from crossdomain.amazonloader import product_partition

if __name__ == '__main__':
    
    print 'Loading Data...'
    json_data = open('data/amazon_products_cleaned').read()

    print 'Parsing hierarchy...'
    graphs = get_amazon_graphs('data/AmazonHeirarchy.json')

    n_partitions = 100

    product_partition(json_data, graphs, n_partitions, prefix = 'amazon-data')


    # can call clean_json_socialmedia() before if you don't have a clean version
    socialmedia = open('data/Social_Conversations_AmazonLabel_clean').read()

    _ = get_socialmedia(socialmedia, graphs)
    # everything is labelled!
    # the twitter entries are in twitter.pkl, and the ebay entries are in ebay.pkl




    