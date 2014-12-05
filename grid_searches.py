from classifier import *
from sklearn.grid_search import GridSearchCV

def SVM_grid_search():

    SVM = SVMClassifier(range(1,20), 100)
    SVM.load_data()
    SVM.prune_training_data(600)
    SVM.train()
    clf = GridSearchCV(estimator=SVM.clf, param_grid={'C': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 5, 10, 100, 1000]})

    X = SVM.get_bag_of_ngrams(SVM.reviews)
    y = SVM.labels
    clf.fit(X, y)
    print clf.best_score_
    print clf.best_estimator_.C

    return

def SGD_grid_search():

    SGDc = SGD(range(1,20), 100)
    SGDc.load_data()
    SGDc.prune_training_data(600)
    SGDc.train()
    clf = GridSearchCV(estimator=SGDc.clf, param_grid={'alpha': 10.0**-np.arange(1,5),
                                                      'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge',
                                                               'perceptron','squared_loss','huber', 'epsilon_insensitive']})

    X = SGDc.get_bag_of_ngrams(SGDc.reviews)
    y = SGDc.labels
    clf.fit(X, y)
    print clf.best_score_
    print clf.best_estimator_.alpha
    print clf.best_estimator_.loss

    return

