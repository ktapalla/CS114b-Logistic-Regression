# CS114B Spring 2023
# Logistic Regression Classifier

import os
import numpy as np
from collections import defaultdict
from math import ceil
from random import Random

import scipy
from scipy.special import expit # logistic (sigmoid) function

class LogisticRegression():

    def __init__(self):
        self.class_dict = {}
        # use of self.feature_dict is optional for this assignment
        self.feature_dict = {}
        self.n_features = None
        self.theta = None # weights (and bias)

    '''
    Given a training set, fills in self.class_dict (and optionally,
    self.feature_dict), as in HW1.
    Also sets the number of features self.n_features and initializes the
    parameter vector self.theta.
    '''
    def make_dicts(self, train_set):
        self.class_dict['neg'] = 0
        self.class_dict['pos'] = 1
        token_ind_count = 0
        # iterate over training documents
        for root, dirs, files in os.walk(train_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    # each line/sentence in a file
                    for line in f:
                        # each word/token/feature in each line
                        split_line = line.split()
                        for token in split_line:
                            # check if token was already seen, if not add to feature_dict
                            if token not in self.feature_dict:
                                self.feature_dict[token] = token_ind_count
                                token_ind_count += 1
                    # fill in class_dict, (feature_dict,) n_features, and theta
                self.n_features = len(self.feature_dict)
                self.theta = np.zeros(self.n_features + 1)


    '''
    Loads a dataset. Specifically, returns a list of filenames, and dictionaries
    of classes and documents such that:
    classes[filename] = class of the document
    documents[filename] = feature vector for the document (use self.featurize)
    '''
    def load_data(self, data_set):
        filenames = []
        classes = dict()
        documents = dict()
        # iterate over documents
        for root, dirs, files in os.walk(data_set):
            for name in files:
                with open(os.path.join(root, name)) as f:
                    doc_arr = set()
                    # features = dict()
                    filenames.append(name)
                    if 'pos' in root:
                        classes[name] = self.class_dict['pos']
                    if 'neg' in root:
                        classes[name] = self.class_dict['neg']
                    # each line/sentence in a file
                    for line in f:
                        # each word/token/feature in each line
                        split_line = line.split()
                        for token in split_line:
                            # doc_arr.append(token)
                            if token in self.feature_dict:
                                doc_arr.add(token)
                    documents[name] = self.featurize(doc_arr)
        return filenames, classes, documents

    '''
    Given a document (as a list of words), returns a feature vector.
    Note that the last element of the vector, corresponding to the bias, is a
    "dummy feature" with value 1.
    '''
    def featurize(self, document):
        vector = np.zeros(self.n_features + 1)
        # your code here
        for word in document:
            word_ind = self.feature_dict[word]
            vector[word_ind] = 1
        vector[-1] = 1
        return vector

    '''
    Trains a logistic regression classifier on a training set.
    '''
    def train(self, train_set, batch_size=3, n_epochs=1, eta=0.1):
        filenames, classes, documents = self.load_data(train_set)
        filenames = sorted(filenames)
        n_minibatches = ceil(len(filenames) / batch_size)
        for epoch in range(n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
            loss = 0
            for i in range(n_minibatches):
                # list of filenames in minibatch
                minibatch = filenames[i * batch_size: (i + 1) * batch_size]
                mb_size = len(minibatch)
                # create and fill in matrix x and vector y
                y = np.zeros(mb_size)
                mx = np.zeros((mb_size, self.n_features + 1))
                for mb_ind in range(mb_size):
                    file = minibatch[mb_ind]
                    mx[mb_ind] = documents[file]
                    y[mb_ind] = classes[file]
                # compute y_hat
                y_hat = scipy.special.expit((np.dot(mx, self.theta)))
                # update loss
                loss += -np.sum(((y * np.log(y_hat)) + ((1 - y) * np.log(1 - y_hat))))
                # compute gradient
                gradient = np.dot(mx.T, (y_hat - y)) / mb_size
                # update weights (and bias)
                self.theta = self.theta - eta * gradient
            loss /= len(filenames)
            print("Average Train Loss: {}".format(loss))
            # randomize order
            Random(epoch).shuffle(filenames)

    '''
    Tests the classifier on a development or test set.
    Returns a dictionary of filenames mapped to their correct and predicted
    classes such that:
    results[filename]['correct'] = correct class
    results[filename]['predicted'] = predicted class
    '''
    def test(self, dev_set):
        results = defaultdict(dict)
        filenames, classes, documents = self.load_data(dev_set)
        # get most likely class (recall that P(y=1|x) = y_hat)
        # iterate over documents
        for file in filenames:
            doc_feats = documents[file]
            corr_class = classes[file]
            # compute y_hat
            y_hat = scipy.special.expit((np.dot(doc_feats, self.theta)))
            if y_hat > 0.5:
                pred_class = 1
            else:
                pred_class = 0
            results[file]['correct'] = corr_class
            results[file]['predicted'] = pred_class
        return results

    '''
    Given results, calculates the following:
    Precision, Recall, F1 for each class
    Accuracy overall
    Also, prints evaluation metrics in readable format.
    '''
    def evaluate(self, results):
        # you can copy and paste your code from HW1 here
        # you may find this helpful
        confusion_matrix = np.zeros((len(self.class_dict), len(self.class_dict)), dtype=np.int64)
        # [[neg_pred_neg_corr_count, neg_pred_pos_corr_count],
        # [pos_pred_neg_corr_count, pos_pred_pos_corr_count]]
        for file in results:
            correct = results[file]['correct']
            predicted = results[file]['predicted']
            if correct == predicted:
                if predicted == 0:
                    confusion_matrix[0][0] += 1
                if predicted == 1:
                    confusion_matrix[1][1] += 1
            else:
                if predicted == 0:
                    confusion_matrix[0][1] += 1
                if predicted == 1:
                    confusion_matrix[1][0] += 1
        # precision, recall, f1 calculations for pos and neg classes, overall accuracy calculations
        neg_precision = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[0][1])
        pos_precision = confusion_matrix[1][1] / (confusion_matrix[1][0] + confusion_matrix[1][1])
        neg_recall = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
        pos_recall = confusion_matrix[1][1] / (confusion_matrix[1][1] + confusion_matrix[0][1])
        neg_f1 = (2 * neg_precision * neg_recall) / (neg_precision + neg_recall)
        pos_f1 = (2 * pos_precision * pos_recall) / (pos_precision + pos_recall)
        true_neg_pos = confusion_matrix[0][0] + confusion_matrix[1][1]
        tn_fn_tp_fp = true_neg_pos + confusion_matrix[0][1] + confusion_matrix[1][0]
        accuracy = true_neg_pos / tn_fn_tp_fp
        # printing the confusion matrix
        print()
        print('MATRIX: ')
        print(confusion_matrix)
        # printing precision, recall, f1, and accuracy calculations with proper labels
        print()
        print('NEGATIVE PRECISION = %.4f or %2.2f%%' % (neg_precision, neg_precision * 100))
        print('POSITIVE PRECISION = %.4f or %2.2f%%' % (pos_precision, pos_precision * 100))
        print('NEGATIVE RECALL    = %.4f or %2.2f%%' % (neg_recall, neg_recall * 100))
        print('POSITIVE RECALL    = %.4f or %2.2f%%' % (pos_recall, pos_recall * 100))
        print('NEGATIVE F1        = %.4f or %2.2f%%' % (neg_f1, neg_f1 * 100))
        print('POSITIVE F1        = %.4f or %2.2f%%' % (pos_f1, pos_f1 * 100))
        print('OVERALL ACCURACY   = %.4f or %2.2f%%' % (accuracy, accuracy * 100))


if __name__ == '__main__':
    lr = LogisticRegression()
    # make sure these point to the right directories
    lr.make_dicts('movie_reviews/train')
    # lr.make_dicts('movie_reviews_small/train')
    lr.train('movie_reviews/train', batch_size=3, n_epochs=1, eta=0.1)
    # lr.train('movie_reviews_small/train', batch_size=3, n_epochs=1, eta=0.1)
    results = lr.test('movie_reviews/dev')
    # results = lr.test('movie_reviews_small/test')
    lr.evaluate(results)
