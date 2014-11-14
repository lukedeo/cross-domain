import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys

### ALERT: some products don't have LABELS!!!

def build_training_data(partitions_to_use, total_num_partitions):
    """
    Builds the training data and labels from the partitions indicated in the list
    partitions to use.
    total_num_partitions is the total number of existing partitions

    Returns the training data in the format [reviews, labels]
    """

    FILE_NAME_TEMPLATE = "data/amazon-data-%s-of-%s.pkl"

    # Holds the reviews obtained in the data
    reviews = []
    # Holds the labels for each review
    # NOTE: For the moment we take the first label we have in the product!
    labels = []
    count = 1
    for i in partitions_to_use:
        sys.stdout.write('Importing package %d out of %d \r' % (count, len(partitions_to_use)))
        sys.stdout.flush()

        file_to_open = FILE_NAME_TEMPLATE % (i, total_num_partitions)
        [products, prod_reviews] = pickle.load(open(file_to_open))
        for review in prod_reviews:
            if len(review['labels']) != 0:
                reviews.append(review['text'])
                labels.append(review['labels'][0])
        count += 1
    sys.stdout.write('\nData loaded\n')
    return [reviews, labels]

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

    count = 0
    if twitter:
        tweets = pickle.load(open(TWITTER_FILE))
        for tweet in tweets:
            if len(tweet['labels']) != 0:
                social_items.append(tweet['text'])
                labels.append(tweet['labels'][0])
                count += 1
    if ebay:
        products = pickle.load(open(EBAY_FILE))
        for product in products:
            if len(product['labels']) != 0:
                social_items.append(product['text'])
                labels.append(product['labels'][0])
                count += 1
    sys.stdout.write('%d elements loaded\n' % count)
    return [social_items, labels]


def train_naive_bayes(reviews, labels, test_reviews, test_labels):

    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(reviews)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, labels)

    print ("Trained!")

    X_test_counts = count_vect.transform(test_reviews)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    success_rate = np.mean(predicted == test_labels)

    print "Success rate: " + str(success_rate)
    return success_rate

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    negative = y == 0
    positive = ~negative
    return prob[negative, 0].sum() + prob[positive, 1].sum()

def train_stochastic_gradient_descent(reviews, labels, test_reviews, test_labels):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(reviews)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    clf = SGDClassifier(loss="hinge", penalty="l2").fit(X_train_tfidf, labels)

    print ("Trained!")

    X_test_counts = count_vect.transform(test_reviews)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    success_rate = np.mean(predicted == test_labels)

    print "Success rate: " + str(success_rate)
    return success_rate

def train_logistic(reviews, labels, test_reviews, test_labels):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(reviews)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    clf = LogisticRegression(C=1e5).fit(X_train_tfidf, labels)

    print ("Trained!")

    X_test_counts = count_vect.transform(test_reviews)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    success_rate = np.mean(predicted == test_labels)

    print "Success rate: " + str(success_rate)
    return success_rate

def train_kNN(reviews, labels, test_reviews, test_labels):
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(reviews)
    tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tfidf = tfidf_transformer.transform(X_train_counts)
    clf = KNeighborsClassifier(n_neighbors=7).fit(X_train_tfidf, labels) # random # for now


    print ("Trained!")

    X_test_counts = count_vect.transform(test_reviews)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    predicted = clf.predict(X_test_tfidf)
    predicted_probs = clf.predict_proba(X_test) # can also use log_proba
    success_rate = np.mean(predicted == test_labels)

    print "Success rate: " + str(success_rate)
    return success_rate


def usage_example():

    # Must have 100 partitions of the data!
    # Partition packages must be on folder data!
    [reviews, labels] = build_training_data(range(1, 71), 100)
    [test_reviews, test_labels] = build_training_data(range(71,101), 100)

    train_naive_bayes(reviews, labels, test_reviews, test_labels)

def load_data():
    [reviews, labels] = build_training_data(range(1, 2), 100);
    [test_reviews, test_labels] = build_training_data(range(2, 4), 100);
    return reviews, labels, test_reviews, test_labels

reviews, labels, test_reviews, test_labels = load_data()


def usage_example_2():
    [reviews, labels] = build_training_data(range(1, 71), 100)
    [test_reviews, test_labels] = build_training_data(range(71,101), 100)
    vectorizers = [TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english',), CountVectorizer()]
    for vect in vectorizers:
        print vect
        train_stochastic_gradient_descent(vect, reviews, labels, test_reviews, test_labels)



def example_crossdomain():
    [reviews, labels] = build_training_data(range(1,41), 100)
    [social_items, social_labels] = build_social_data()

    train_naive_bayes(reviews, labels, social_items, social_labels)
