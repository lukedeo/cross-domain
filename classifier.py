import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import sys

class CrossDomainClassifier(object):

    FILE_NAME_TEMPLATE = "data/amazon-data-%s-of-%s.pkl"
    TWITTER_FILE = 'data/twitter.pkl'
    EBAY_FILE = 'data/ebay.pkl'

    def __init__(self, partitions_to_use, total_num_partitions):
        self.partitions_to_use = partitions_to_use
        self.total_num_partitions = total_num_partitions

    @staticmethod
    def load_training_data(partitions_to_use, total_num_partitions):
        """
        Builds the training data and labels from the partitions indicated in the list
        partitions to use.
        total_num_partitions is the total number of existing partitions
        """

        # Holds the reviews obtained in the data
        reviews = []
        # Holds the labels for each review
        # NOTE: For the moment we take the first label we have in the product!
        labels = []
        count = 1
        for i in partitions_to_use:
            sys.stdout.write('Importing package %d out of %d \r' % (count, len(partitions_to_use)))
            sys.stdout.flush()

            file_to_open = CrossDomainClassifier.FILE_NAME_TEMPLATE % (i, total_num_partitions)
            [products, prod_reviews] = pickle.load(open(file_to_open))
            for review in prod_reviews:
                if len(review['labels']) != 0:
                    reviews.append(review['text'])
                    labels.append(review['labels'][0])
            count += 1
        sys.stdout.write('\nData loaded\n')

        return [reviews, labels]

    def __load_cross_domain_data(self):
        """
        Builds the twitter data and gets the labels of each item. Allows retrieving from
        differents sources.

        Returns the data in the format [social_items, label]
        """

        # Holds the social items (tweets, ebay reviews...)
        social_items = []
        # Holds the labels for each social item
        # NOTE: For the moment we take the first label we have in the product!
        labels = []

        count = 0
        tweets = pickle.load(open(self.TWITTER_FILE))
        for tweet in tweets:
            if len(tweet['labels']) != 0:
                social_items.append(tweet['text'])
                labels.append(tweet['labels'][0])
                count += 1
        tweets = pickle.load

        self.twitter_items = social_items
        self.twitter_labels = labels

        social_items = []
        labels = []
        products = pickle.load(open(self.EBAY_FILE))
        for product in products:
            if len(product['labels']) != 0:
                social_items.append(product['text'])
                labels.append(product['labels'][0])
                count += 1

        self.ebay_items = social_items
        self.ebay_labels = labels

    def load_data(self):
        self.reviews, self.labels = self.load_training_data(self.partitions_to_use, self.total_num_partitions)
        self.__load_cross_domain_data()

    def load_test_data(self, partitions_to_use):
        """
        Loads test data into the object
        """
        self.test_reviews, self.test_labels = self.load_training_data(partitions_to_use, self.total_num_partitions)

    def get_data(self):
        """
        Returns training and label data
        """
        return [self.reviews, self.labels]

    def get_cross_domain_data(self):
        return {'twitter': [self.twitter_items, self.twitter_labels],
                'ebay': [self.ebay_items, self.ebay_labels]}

    def train(self):
        raise NotImplementedError()

    def __test(self, reviews, labels):
        raise NotImplementedError()

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}


class NaiveBayesClassifier(CrossDomainClassifier):
    """
    Naive bayes classifier with tfidf
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(self.reviews)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        self.clf = MultinomialNB().fit(X_train_tfidf, self.labels)

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}


class SGD(CrossDomainClassifier):
    """
    Stochastic Gradient Descent with Tfidf
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(self.reviews)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        self.clf = SGDClassifier(loss="hinge", penalty="l2").fit(X_train_tfidf, self.labels)

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}

class LogisticClassifier(CrossDomainClassifier):
    """
    Logistic Regression with Tfidf
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(self.reviews)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        self.clf = LogisticRegression(C=1e5).fit(X_train_tfidf, self.labels)

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}

class kNNClassifier(CrossDomainClassifier):
    """
    K nearest neighbors with Tfidf
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(self.reviews)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        self.clf = KNeighborsClassifier(n_neighbors=7).fit(X_train_tfidf, self.labels)

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}



def example():
    NB = NaiveBayesClassifier(range(1,10), 100) # Select which partitions we are going to use
    NB.load_data() # Actually load the data from the partitions
    NB.train()
    train_error = NB.get_training_error()
    print ("Training error: " + str(train_error))

    # Load some test data and get error
    NB.load_test_data(range(10,13))
    generalized_error = NB.get_generalized_error()
    print ("Test erorr: " + str(generalized_error))

    # Cross domain classification error
    cs_error = NB.get_crossdomain_error()
    print ("Twitter data error: " + str(cs_error['twitter']))
    print ("Ebay data error: " + str(cs_error['ebay']))

