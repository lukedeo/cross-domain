#!/usr/bin/env python

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


    