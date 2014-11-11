import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
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

    # we can check if overfit/underfit
    training_accuracy = 100 * clf.score(xtrain, ytrain)
    test_accuracy = 100 * clf.score(xtest, ytest)

    print "Success rate: " + str(success_rate)
    return success_rate

def log_likelihood(clf, x, y):
    prob = clf.predict_log_proba(x)
    negative = y == 0
    positive = ~negative
    return prob[negative, 0].sum() + prob[positive, 1].sum()

def train_stochastic_gradient_descent(vectorizer, reviews, labels, test_reviews, test_labels):
    X_train = vectorizer.fit_transform(reviews)
    clf = SGDClassifier(loss="hinge", penalty="l2").fit(X_train)
    print clf
    X_test = vectorizer.transform(test_reviews)
    predicted = clf.predict(X_test)

    training_accuracy = 100 * clf.score(xtrain, ytrain)
    test_accuracy = 100 * clf.score(xtest, ytest)

    print "Success rate: " + str(test_accuracy) + "%"
    return test_accuracy


vectorizers = [TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english',), CountVectorizer()]
   


def usage_example():

    # Must have 100 partitions of the data!
    # Partition packages must be on folder data!
    [reviews, labels] = build_training_data(range(1, 71), 100)
    [test_reviews, test_labels] = build_training_data(range(71,101), 100)

    train_naive_bayes(reviews, labels, test_reviews, test_labels)