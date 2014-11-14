"""
Functions to get insight into the data
"""
import sys
import pickle


#
# Categories anaylisis of all the amazon data
#
# Number of products: 2498330
# Multilabel elements: 888864
# Percentage of products with a given category
# ============================================
# Collectibles: 0.000273
# Music: 0.024316
# VideoGames: 0.019782
# Electronics: 0.079275
# Beauty: 0.020367
# Automotive: 0.057635
# Movies & TV: 0.000462
# no_category: 0.016674
# Baby: 0.017930
# Books: 0.408854
# Kitchen: 0.083820
# Everything Else: 0.000734
# Grocery: 0.018467
# MobileApps: 0.000008
# Software: 0.004045
# KindleStore: 0.275891
# SportingGoods: 0.090299
# OfficeProducts: 0.032052
# ArtsAndCrafts: 0.017305
# Magazines: 0.009083
# Appliances: 0.007523
# Toys: 0.029429
# LawnAndGarden: 0.026913
# Tools: 0.051303
# MusicalInstruments: 0.022971
# HealthPersonalCare: 0.047808

def categories_distribution(partitions_to_use, total_num_partitions):
    """
    Gives information about the frequency of categories or number of elements with more than one category
    Gets the data from a list of partitions of data
    """

    FILE_NAME_TEMPLATE = "data/amazon-data-%s-of-%s.pkl"

    multilabel_count = 0
    cat_freq = {'no_category': 0}
    num_products = 0
    count = 1
    for i in partitions_to_use:
        sys.stdout.write('Analyzing package %d out of %d \r' % (count, len(partitions_to_use)))
        sys.stdout.flush()

        file_to_open = FILE_NAME_TEMPLATE % (i, total_num_partitions)
        [products, prod_reviews] = pickle.load(open(file_to_open))

        for review in prod_reviews:
            labels = review['labels']

            # Count categories, and number of products with more than one label
            if len(labels) == 0:
                cat_freq['no_category'] += 1
            else:
                if len(labels) > 1:
                    multilabel_count += 1
                for cat in labels:
                    if cat in cat_freq:
                        cat_freq[cat] += 1
                    else:
                        cat_freq[cat] = 1

            num_products += 1

            # Just in case we need to get the data afterwards
            # if len(review['labels']) != 0:
            #     reviews.append(review['text'])
            #     labels.append(review['labels'][0])

        count += 1

    #Normalize data
    for cat in cat_freq:
        cat_freq[cat] = 1.0 * cat_freq[cat] / num_products

    # Show data
    sys.stdout.write("\nNumber of products: %d" % num_products)
    sys.stdout.write("\nMultilabel elements: %d \n" % multilabel_count)
    sys.stdout.write("Percentage of products with a given category\n")
    sys.stdout.write("============================================\n")

    for cat in cat_freq:
        sys.stdout.write("%s: %f\n" % (cat, cat_freq[cat]))
    sys.stdout.write("")

    return cat_freq



def build_social_data(twitter=True, ebay=True):
    """
    Builds the twitter data and gets the labels of each item. Allows retrieving from
    differents sources.

    Returns the data in the format [social_items, label]
    """

    TWITTER_FILE = 'data/twitter.pkl'
    EBAY_FILE = 'data/ebay.pkl'

    # Holds the social items (tweets, ebay reviews...)
    social_items = []
    # Holds the labels for each social item
    # NOTE: For the moment we take the first label we have in the product!
    labels = []


    multilabel_count = 0
    cat_freq = {'no_category': 0}
    num_products = 0
    count = 1
    count = 0
    if twitter:
        tweets = pickle.load(open(TWITTER_FILE))
        for tweet in tweets:
            labels = tweet['labels']
            # Count categories, and number of products with more than one label
            if len(labels) == 0:
                cat_freq['no_category'] += 1
            else:
                if len(labels) > 1:
                    multilabel_count += 1
                for cat in labels:
                    if cat in cat_freq:
                        cat_freq[cat] += 1
                    else:
                        cat_freq[cat] = 1

            num_products += 1

    if ebay:
        products = pickle.load(open(EBAY_FILE))
        for product in products:
            labels = product['labels']
            # Count categories, and number of products with more than one label
            if len(labels) == 0:
                cat_freq['no_category'] += 1
            else:
                if len(labels) > 1:
                    multilabel_count += 1
                for cat in labels:
                    if cat in cat_freq:
                        cat_freq[cat] += 1
                    else:
                        cat_freq[cat] = 1

            num_products += 1

    #Normalize data
    for cat in cat_freq:
        cat_freq[cat] = 1.0 * cat_freq[cat] / num_products

    # Show data
    sys.stdout.write("\nNumber of products: %d" % num_products)
    sys.stdout.write("\nMultilabel elements: %d \n" % multilabel_count)
    sys.stdout.write("Percentage of products with a given category\n")
    sys.stdout.write("============================================\n")

    for cat in cat_freq:
        sys.stdout.write("%s: %f\n" % (cat, cat_freq[cat]))
    sys.stdout.write("")

    return cat_freq



    sys.stdout.write('%d elements loaded\n' % count)
    return [social_items, labels]


def categories_distribution(labels):
    """
    Gives information about the frequency of categories or number of elements with more than one category
    Gets the data from a list of labels
    """
    multilabel_count = 0
    cat_freq = {'no_category': 0}
    num_products = 0
    count = 1
    for i in partitions_to_use:
        sys.stdout.write('Analyzing package %d out of %d \r' % (count, len(partitions_to_use)))
        sys.stdout.flush()

        file_to_open = FILE_NAME_TEMPLATE % (i, total_num_partitions)
        [products, prod_reviews] = pickle.load(open(file_to_open))

        for review in prod_reviews:
            labels = review['labels']

            # Count categories, and number of products with more than one label
            if len(labels) == 0:
                cat_freq['no_category'] += 1
            else:
                if len(labels) > 1:
                    multilabel_count += 1
                for cat in labels:
                    if cat in cat_freq:
                        cat_freq[cat] += 1
                    else:
                        cat_freq[cat] = 1

            num_products += 1

            # Just in case we need to get the data afterwards
            # if len(review['labels']) != 0:
            #     reviews.append(review['text'])
            #     labels.append(review['labels'][0])

        count += 1

    #Normalize data
    for cat in cat_freq:
        cat_freq[cat] = 1.0 * cat_freq[cat] / num_products

    # Show data
    sys.stdout.write("\nNumber of products: %d" % num_products)
    sys.stdout.write("\nMultilabel elements: %d \n" % multilabel_count)
    sys.stdout.write("Percentage of products with a given category\n")
    sys.stdout.write("============================================\n")

    for cat in cat_freq:
        sys.stdout.write("%s: %f\n" % (cat, cat_freq[cat]))
    sys.stdout.write("")

    return cat_freq







