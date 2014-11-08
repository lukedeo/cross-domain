#!/usr/bin/env python

import crossdomain as domain

if __name__ == '__main__':
    
    print 'Loading Data...'
    json_data = open('project/data/amazon_products_cleaned').read()

    print 'Parsing hierarchy...'
    graphs = domain.get_amazon_graphs('project/data/AmazonHeirarchy.json')

    n_partitions = 1000

    domain.product_partition(json_data, graphs, n_partitions, prefix = 'project/amazon-data')