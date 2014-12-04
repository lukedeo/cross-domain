import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


# Complete list of Amazon categories (without KindleStore ebooks)
# Can be useful for reference
categories = [u'SportingGoods',
              u'Books',
              u'Magazines',
              u'Music',
              u'Automotive',
              u'Baby',
              u'Electronics',
              u'Toys',
              u'Tools',
              u'HealthPersonalCare',
              u'Beauty',
              u'VideoGames',
              u'OfficeProducts',
              u'MusicalInstruments',
              u'LawnAndGarden',
              u'ArtsAndCrafts',
              u'Kitchen',
              u'Grocery',
              u'Appliances',
              u'Everything Else',
              u'Movies & TV',
              u'Software',
              u'Collectibles',
              u'MobileApps']

def get_categories_stats(labels, display_info=True):
    """
    Given a set of labels / categories, counts the elements of each category and
    provides the relative appearance of the category in the dataset
    """
    categories = []
    for label in labels:
        if label not in categories:
            categories.append(label)

    categories_count = dict.fromkeys(categories)
    total_count = len(labels)
    for cat in categories:
        categories_count[cat] = labels.count(cat)
        if display_info:
            print ("%s: %d elements, %f" % (cat, categories_count[cat], categories_count[cat] * 100.0 / total_count))

    return categories_count


def get_learning_curve(classifier, range=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]):

    # Balance classes
    stats = get_categories_stats(classifier.labels, display_info=False)
    max_value = max(stats.values())

    # Prune to 0.1 * max_class_elements (this also fully randomizes data!)
    classifier.prune_training_data(max_value / 10)

    data_size = []
    training_scores = []
    test_scores = []
    twitter_scores = []
    ebay_scores = []
    num_iter = 0
    for i in range:
        num_iter += 1
        print "Iteration %d of %d" % (num_iter, len(range))

        num_elems = int(len(classifier.reviews) * i)
        classifier.train(num_elems)
        _, _, f_score_training, _ = classifier.get_scores_training()
        _, _, f_score_test, _ = classifier.get_scores_test()
        _, _, f_score_twitter, _ = classifier.get_scores_twitter()
        _, _, f_score_ebay, _ = classifier.get_scores_ebay()
        data_size.append(num_elems)
        training_scores.append(f_score_training)
        test_scores.append(f_score_test)
        twitter_scores.append(f_score_twitter)
        ebay_scores.append(f_score_ebay)

    plt.figure()
    plt.title("Learning curve")
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("F1 score")
    plt.grid()

    plt.plot(data_size, training_scores, 'o-', color="r",
             label="Training score")
    plt.plot(data_size, test_scores, 'o-', color="g",
             label="Test score")
    plt.plot(data_size, twitter_scores, 'o-', color="b",
             label="Cross-domain: twitter")
    plt.plot(data_size, ebay_scores, 'o-', color="y",
             label="Cross-domain: eBay")

    plt.legend(loc="best")
    plt.show()


def plot_confusion_matrix(cm):
    """
    Plots a scikit learn confusion matrix
    """
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True category')
    plt.xlabel('Predicted category')
    plt.show()

class CrossDomainClassifier(object):

    FILE_NAME_TEMPLATE = "data/amazon-data-%s-of-%s.pkl"
    TWITTER_FILE = 'data/twitter.pkl'
    EBAY_FILE = 'data/ebay.pkl'

    def __init__(self, partitions_to_use, total_num_partitions, ngram_range=(1,1)):
        self.partitions_to_use = partitions_to_use
        self.total_num_partitions = total_num_partitions
        self.ngram_range = ngram_range

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
                    label = review['labels'][random.choice(range(len(review['labels'])))] # Choose at random
                    if label == u'KindleStore': # Kindle and books are the same entity
                        label = u'Books'
                    labels.append(label)
            count += 1
        sys.stdout.write('\nData loaded\n')

        return [reviews, labels]

    @staticmethod
    def __prune_data(reviews, labels, max_elements_by_category):
        """ Balances given categories to a max number of elements so that more data can be processed or in order
        to balance classes. Removal of elements is random"""

        # Randomly shuffle lists of reviews and labels so that deletion is random
        reviews_shuf = []
        labels_shuf = []
        index_shuf = range(len(reviews))
        random.shuffle(index_shuf)
        for i in index_shuf:
            reviews_shuf.append(reviews[i])
            labels_shuf.append(labels[i])

        # Add categories and reviews until max is obtained or there are no more elements
        reviews_result = []
        labels_result = []
        categories_count = get_categories_stats(labels, display_info=False)
        for cat in categories_count:
            objective_count = min(categories_count[cat], max_elements_by_category)
            i = 0
            elements_added = 0
            while elements_added != objective_count:
                if labels_shuf[i] == cat:
                    labels_result.append(labels_shuf[i])
                    reviews_result.append(reviews_shuf[i])
                    elements_added += 1
                i += 1

        # Shuffle again the labels and reviews
        reviews_shuf = []
        labels_shuf = []
        index_shuf = range(len(reviews_result))
        random.shuffle(index_shuf)
        for i in index_shuf:
            reviews_shuf.append(reviews_result[i])
            labels_shuf.append(labels_result[i])

        return (reviews_shuf, labels_shuf)

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
                label = tweet['labels'][random.choice(range(len(tweet['labels'])))]
                if label == u'KindleStore': # Kindle and books are the same entity
                    label = u'Books'
                labels.append(label)
                count += 1

        self.twitter_items = social_items
        self.twitter_labels = labels

        social_items = []
        labels = []
        products = pickle.load(open(self.EBAY_FILE))
        for product in products:
            if len(product['labels']) != 0:
                social_items.append(product['text'])
                label = product['labels'][random.choice(range(len(product['labels'])))]
                if label == u'KindleStore': # Kindle and books are the same entity
                    label = u'Books'
                labels.append(label)
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

    def prune_training_data(self, max_elements_by_category):
        """
        Prunes training data so that classes are more balanced
        """
        self.reviews, self.labels = self.__prune_data(self.reviews, self.labels, max_elements_by_category)

    def prune_test_data(self, max_elements_by_category):
        """
        Prunes test data so that classes are more balanced
        """
        self.test_reviews, self.test_labels = self.__prune_data(self.test_reviews, self.test_labels, max_elements_by_category)

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

    def get_bag_of_ngrams(self, texts, ngram_range=None):
        """ Sets vectorizer feature and returns data from object in featuer form X """
        if ngram_range is None:
            ngram_range = self.ngram_range
        self.count_vect = CountVectorizer(ngram_range=ngram_range)
        X_train_counts = self.count_vect.fit_transform(texts)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)

        return X_train_tfidf


class NaiveBayesClassifier(CrossDomainClassifier):
    """
    Naive bayes classifier with tfidf
    """

    def train(self, limit_data=None):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        if limit_data is None:
            limit_data = len(self.reviews)

        X = self.get_bag_of_ngrams(self.reviews[:limit_data])
        self.clf = MultinomialNB().fit(X, self.labels[:limit_data])

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        self.cm = confusion_matrix(labels, predicted)

        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}

    def __get_scores(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        return precision_recall_fscore_support(labels, predicted, average='macro')

    def get_scores_training(self):
        return self.__get_scores(self.reviews, self.labels)

    def get_scores_test(self):
        return self.__get_scores(self.test_reviews, self.test_labels)

    def get_scores_twitter(self):
        return self.__get_scores(self.twitter_items, self.twitter_labels)

    def get_scores_ebay(self):
        return self.__get_scores(self.ebay_items, self.ebay_labels)


class SGD(CrossDomainClassifier):
    """
    Stochastic Gradient Descent with Tfidf
    """
    def train(self, limit_data=None):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        if limit_data is None:
            limit_data = len(self.reviews)

        X = self.get_bag_of_ngrams(self.reviews[:limit_data])
        self.clf = SGDClassifier(loss="hinge", penalty="l2").fit(X, self.labels[:limit_data])

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        self.cm = confusion_matrix(labels, predicted)

        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}

    def __get_scores(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        return precision_recall_fscore_support(labels, predicted, average='macro')

    def get_scores_training(self):
        return self.__get_scores(self.reviews, self.labels)

    def get_scores_test(self):
        return self.__get_scores(self.test_reviews, self.test_labels)

    def get_scores_twitter(self):
        return self.__get_scores(self.twitter_items, self.twitter_labels)

    def get_scores_ebay(self):
        return self.__get_scores(self.ebay_items, self.ebay_labels)


class LogisticClassifier(CrossDomainClassifier):
    """
    Logistic Regression with Tfidf
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        X = self.get_bag_of_ngrams(self.reviews)
        self.clf = LogisticRegression(C=1e5).fit(X, self.labels)

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        self.cm = confusion_matrix(labels, predicted)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}

class PerceptronClassifier(CrossDomainClassifier):
    """
    Perceptron Classifier with TFIDF
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        self.count_vect = CountVectorizer()
        X_train_counts = self.count_vect.fit_transform(self.reviews)
        self.tfidf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
        X_train_tfidf = self.tfidf_transformer.transform(X_train_counts)
        self.clf = Perceptron(n_iter=30, shuffle=False).fit(X_train_tfidf, self.labels)

    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        self.cm = confusion_matrix(labels, predicted)
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
    K nearest neighbors with Tfidf-NOT TO BE USED
    """

    def train(self):
        if not hasattr(self, 'reviews'):
            print "No data loaded"
            return

        X = self.get_bag_of_ngrams(self.reviews)
        self.clf = KNeighborsClassifier(n_neighbors=3).fit(X, self.labels)


    def __test(self, reviews, labels):
        X_training_counts = self.count_vect.transform(reviews)
        X_training_tfidf = self.tfidf_transformer.transform(X_training_counts)

        predicted = self.clf.predict(X_training_tfidf)
        self.cm = confusion_matrix(labels, predicted)
        return 1 - np.mean(predicted == labels)

    def get_training_error(self):
        return self.__test(self.reviews, self.labels)

    def get_generalized_error(self):
        return self.__test(self.test_reviews, self.test_labels)

    def get_crossdomain_error(self):
        return {'twitter': self.__test(self.twitter_items, self.twitter_labels),
                'ebay': self.__test(self.ebay_items, self.ebay_labels)}



def AkuaExample():
    NB = DecisionTreeClassifier(range(1,2), 100) # Select which partitions we are going to use
    NB.load_data() # Actually load the data from the partitions
    NB.train()
    train_error = NB.get_training_error()
    print ("Training error: " + str(train_error))

    # Load some test data and get error
    NB.load_test_data(range(2,3))
    generalized_error = NB.get_generalized_error()
    print ("Test error: " + str(generalized_error))
    plot_confusion_matrix(NB.cm)

    # Cross domain classification error
    cs_error = NB.get_crossdomain_error()
    print ("Twitter data error: " + str(cs_error['twitter']))
    print ("Ebay data error: " + str(cs_error['ebay']))
    




def example():
    NB = NaiveBayesClassifier(range(1,10), 100) # Select which partitions we are going to use
    NB.load_data() # Actually load the data from the partitions
    NB.train()
    train_error = NB.get_training_error()
    print ("Training error: " + str(train_error))

    # Load some test data and get error
    NB.load_test_data(range(10,13))
    generalized_error = NB.get_generalized_error()
    print ("Test error: " + str(generalized_error))

    # Cross domain classification error
    cs_error = NB.get_crossdomain_error()
    print ("Twitter data error: " + str(cs_error['twitter']))
    print ("Ebay data error: " + str(cs_error['ebay']))

